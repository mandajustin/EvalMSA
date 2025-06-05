from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Literal
from Bio import AlignIO
import tempfile
import subprocess
import os
import psutil
import uuid
import json
import time
import traceback
import threading
from evaluator import calculate_entropy, calculate_blosum_score, calculate_gap_fraction

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported tools - replaced dalign with probcons and kalign
TOOLS = ["mafft", "clustalo", "muscle", "t_coffee", "probcons", "kalign", "prank"]

# Request model - updated with new tools
class EvaluationRequest(BaseModel):
    sequence: str
    email: EmailStr
    programs: List[Literal["mafft", "clustalo", "muscle", "t_coffee", "probcons", "kalign", "prank"]]

# Response models
class EvalResult(BaseModel):
    tool: str
    blosum_score: int
    entropy: float
    gap_fraction: float
    cpu_time_sec: float
    memory_usage_mb: float

class MultiToolResult(BaseModel):
    session_id: str
    results: List[EvalResult]

@app.post("/evaluate", response_model=MultiToolResult)
async def evaluate_from_text(request: EvaluationRequest):
    print("Received Evaluation Request:")
    print(f"Email: {request.email}")
    print(f"Sequence: {request.sequence[:100]}...")
    print(f"Programs: {request.programs}")

    if not request.sequence or not request.programs:
        raise HTTPException(status_code=400, detail="Sequence and programs are required.")

    session_id = str(uuid.uuid4())
    results = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.fasta")
            with open(input_path, "w") as f:
                f.write(request.sequence)

            for tool in request.programs:
                output_path = os.path.join(tmpdir, f"aligned_{tool}.fasta")

                if tool == "mafft":
                    cmd = ["mafft", "--auto", input_path]
                    write_stdout = True
                    timeout = 180
                elif tool == "clustalo":
                    cmd = ["clustalo", "-i", input_path, "-o", output_path, "--outfmt", "fasta", "--force"]
                    write_stdout = False
                    timeout = 180
                elif tool == "muscle":
                    cmd = ["muscle", "-align", input_path, "-output", output_path]
                    write_stdout = False
                    timeout = 180
                elif tool == "t_coffee":
                    cmd = ["t_coffee", input_path, "-output", "fasta_aln", "-outfile", output_path]
                    write_stdout = False
                    timeout = 300  # T-Coffee often needs more time
                elif tool == "probcons":
                    # ProbCons reads from input and writes to stdout
                    cmd = ["probcons", input_path]
                    write_stdout = True
                    timeout = 240
                elif tool == "kalign":
                    # Kalign can write to stdout or use -o flag
                    cmd = ["kalign", "-o", output_path, input_path]
                    write_stdout = False
                    timeout = 180
                elif tool == "prank":
                    # PRANK command - may need adjustment based on version
                    cmd = ["prank", "-d=" + input_path, "-o=" + output_path.replace('.fasta', '')]
                    write_stdout = False
                    timeout = 300  # PRANK can be slow for complex alignments
                    # PRANK typically outputs with .best.fas extension
                    expected_prank_output = output_path.replace('.fasta', '.best.fas')
                else:
                    continue

                # Start subprocess and monitor memory and CPU
                start_time = time.time()
                wall_time_start = time.process_time()  # Get CPU time before process starts
                
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE if write_stdout else subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                
                # Monitor memory and CPU usage in background
                mem_usage_list = []
                cpu_time_samples = []
                
                def monitor_process_usage(proc, mem_list, cpu_samples):
                    try:
                        p = psutil.Process(proc.pid)
                        last_cpu_times = None
                        
                        while proc.poll() is None:
                            try:
                                # Memory tracking
                                mem = p.memory_info().rss / (1024 * 1024)  # MB
                                mem_list.append(mem)
                                
                                # CPU tracking - accumulate CPU time deltas
                                current_times = p.cpu_times()
                                if last_cpu_times is not None:
                                    # Calculate delta between samples
                                    user_delta = current_times.user - last_cpu_times.user
                                    system_delta = current_times.system - last_cpu_times.system
                                    cpu_samples.append((user_delta, system_delta))
                                
                                last_cpu_times = current_times
                                
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                break
                            
                            time.sleep(0.1)  # Sample every 100ms
                    except Exception as e:
                        print(f"Process monitoring error: {e}")
                
                # Start the monitoring thread with both memory and CPU tracking
                monitor_thread = threading.Thread(
                    target=monitor_process_usage, 
                    args=(proc, mem_usage_list, cpu_time_samples)
                )
                monitor_thread.start()

                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                    monitor_thread.join()
                except subprocess.TimeoutExpired:
                    proc.kill()
                    monitor_thread.join()
                    raise HTTPException(status_code=500, detail=f"{tool} timed out after {timeout} seconds.")

                end_time = time.time()
                wall_time_end = time.process_time()
                
                # Calculate CPU time
                wall_cpu_time = wall_time_end - wall_time_start
                
                # Sum up all the CPU time deltas we collected
                user_cpu_time = sum([sample[0] for sample in cpu_time_samples]) if cpu_time_samples else 0.0
                system_cpu_time = sum([sample[1] for sample in cpu_time_samples]) if cpu_time_samples else 0.0
                
                # Fallback to wall time calculation if we didn't get psutil samples
                cpu_time = user_cpu_time + system_cpu_time
                
                # If we still have no CPU time, fall back to wall time with a scaling factor
                # This is an estimation that assumes reasonable CPU usage for MSA tools (typically CPU-bound)
                if cpu_time < 0.01:  # Very small values likely indicate tracking failure
                    elapsed_time = end_time - start_time
                    cpu_time = elapsed_time * 0.9  # Assume 90% CPU usage as fallback

                # Handle tool-specific output processing
                if tool in ["mafft", "probcons"] and write_stdout and stdout:
                    with open(output_path, "wb") as f:
                        f.write(stdout)
                elif tool == "prank":
                    # PRANK creates output with .best.fas extension
                    if os.path.exists(expected_prank_output):
                        # Rename to expected output path
                        os.rename(expected_prank_output, output_path)
                    else:
                        # Check for other possible PRANK output files
                        prank_base = output_path.replace('.fasta', '')
                        possible_outputs = [
                            f"{prank_base}.best.fas",
                            f"{prank_base}.best.fasta",
                            f"{prank_base}.fas",
                            f"{prank_base}.fasta"
                        ]
                        found_output = False
                        for possible_file in possible_outputs:
                            if os.path.exists(possible_file):
                                os.rename(possible_file, output_path)
                                found_output = True
                                break
                        if not found_output:
                            print(f"PRANK stderr: {stderr.decode() if stderr else 'No stderr'}")
                            raise HTTPException(status_code=500, detail=f"PRANK did not produce expected output file")
                
                # Print stderr for debugging if there are issues
                if stderr and len(stderr) > 0:
                    print(f"{tool} stderr: {stderr.decode()}")

                # Validate output and check for empty sequences
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise HTTPException(status_code=500, detail=f"{tool} did not produce output.")
                
                # Additional validation: check if file has meaningful content
                try:
                    with open(output_path, 'r') as f:
                        content = f.read()
                        if len(content.strip()) < 10:  # Very short files are likely empty/invalid
                            raise HTTPException(status_code=500, detail=f"{tool} produced empty or invalid output.")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Cannot read {tool} output file: {e}")

                try:
                    # Try different formats for reading alignment files
                    alignment = None
                    formats_to_try = ["fasta", "clustal"]
                    
                    for fmt in formats_to_try:
                        try:
                            alignment = AlignIO.read(output_path, fmt)
                            print(f"Successfully read {tool} output using format: {fmt}")
                            break
                        except Exception as fmt_error:
                            print(f"Failed to read {tool} output with format {fmt}: {fmt_error}")
                            continue
                    
                    if alignment is None:
                        raise Exception("Could not read alignment in any supported format")
                        
                except Exception as e:
                    print(f"Error reading alignment from {tool}: {e}")
                    # Try to read the file content for debugging
                    try:
                        with open(output_path, 'r') as f:
                            content = f.read()
                            print(f"Output file content (first 500 chars): {content[:500]}")
                    except:
                        pass
                    raise HTTPException(status_code=500, detail=f"Failed to read alignment from {tool}: {e}")

                # Log CPU usage info for debugging
                print(f"{tool} CPU measurements:")
                print(f"  - User CPU time: {user_cpu_time:.3f}s")
                print(f"  - System CPU time: {system_cpu_time:.3f}s")
                print(f"  - Total CPU time: {cpu_time:.3f}s")
                print(f"  - Wall time: {end_time - start_time:.3f}s")
                
                result = EvalResult(
                    tool=tool,
                    blosum_score=calculate_blosum_score(alignment),
                    entropy=calculate_entropy(alignment),
                    gap_fraction=calculate_gap_fraction(alignment),
                    cpu_time_sec=cpu_time,  # Now contains accumulated CPU time
                    memory_usage_mb=max(mem_usage_list) if mem_usage_list else 0.0
                )
                results.append(result)

            response = MultiToolResult(session_id=session_id, results=results)

            # Save JSON in temp dir
            tmp_json_path = os.path.join(tmpdir, f"msa_eval_{session_id}.json")
            with open(tmp_json_path, "w") as jf:
                json.dump(response.dict(), jf, indent=4)

            # Save a copy in app root
            app_root_json_path = os.path.join(os.getcwd(), f"msa_eval_{session_id}.json")
            with open(app_root_json_path, "w") as jf:
                json.dump(response.dict(), jf, indent=4)

            return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")