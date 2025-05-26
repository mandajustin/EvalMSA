import os
import subprocess
from pathlib import Path

# Define MSA tools and commands
msa_tools = {
    "clustalo": "clustalo -i {infile} -o {outfile} --force",
    "mafft": "mafft --auto {infile} > {outfile}",
    "muscle": "muscle -align {infile} -output {outfile}",
    "t_coffee": "t_coffee {infile} -output fasta_aln -outfile {outfile}"
}

# Main runner
def run_aligners(input_fasta, output_dir="alignments"):
    os.makedirs(output_dir, exist_ok=True)
    for tool, cmd_template in msa_tools.items():
        tool_dir = Path(output_dir) / tool
        tool_dir.mkdir(parents=True, exist_ok=True)
        output_file = tool_dir / "aligned.fasta"
        cmd = cmd_template.format(infile=input_fasta, outfile=output_file)
        print(f"\nRunning {tool}...")
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"{tool} alignment saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {tool}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run multiple MSA tools on input FASTA")
    parser.add_argument("input_fasta", help="Path to input FASTA file")
    parser.add_argument("--output_dir", default="alignments", help="Directory to store alignment results")
    args = parser.parse_args()
    
    run_aligners(args.input_fasta, args.output_dir)
