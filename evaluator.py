from Bio import AlignIO
from Bio.Align import substitution_matrices
import numpy as np

def calculate_entropy(alignment):
    """Calculate the average entropy per column in the alignment."""
    entropies = []
    for i in range(alignment.get_alignment_length()):
        col = alignment[:, i]
        freq = {aa: col.count(aa) / len(col) for aa in set(col)}
        entropy = -sum(p * np.log2(p) for p in freq.values() if p > 0)
        entropies.append(entropy)
    return float(np.mean(entropies))

def calculate_blosum_score(alignment):
    """Calculate the BLOSUM62 score for the alignment."""
    try:
        matrix = substitution_matrices.load("BLOSUM62")
        total_score = 0
        valid_comparisons = 0
        
        for i in range(len(alignment)):
            for j in range(i + 1, len(alignment)):
                seq_score = 0
                seq_valid_comparisons = 0
                
                for a, b in zip(str(alignment[i].seq), str(alignment[j].seq)):
                    # Skip gap-gap comparisons
                    if a == "-" and b == "-":
                        continue
                    
                    # Handle gap penalties - use milder penalty
                    if a == "-" or b == "-":
                        seq_score -= 2  # Reduced gap penalty
                        seq_valid_comparisons += 1
                        continue
                    
                    # Handle standard amino acids using correct matrix access
                    try:
                        # BLOSUM matrix uses tuple keys like ('A', 'L')
                        score = matrix[(a, b)]
                        seq_score += score
                        seq_valid_comparisons += 1
                    except KeyError:
                        # For non-standard amino acids, use neutral score
                        seq_score += 0  # Changed from -1 to 0
                        seq_valid_comparisons += 1
                
                if seq_valid_comparisons > 0:
                    total_score += seq_score
                    valid_comparisons += seq_valid_comparisons
        
        # Prevent division by zero
        if valid_comparisons == 0:
            print("Warning: No valid amino acid comparisons found for BLOSUM scoring")
            return 0
        
        # Return the average score per comparison
        return total_score / len(alignment) if len(alignment) > 1 else total_score
        
    except Exception as e:
        print(f"Error calculating BLOSUM score: {e}")
        return 0

def calculate_gap_fraction(alignment):
    """Calculate the fraction of gaps in the alignment."""
    total_gaps = sum(str(rec.seq).count("-") for rec in alignment)
    total_length = len(alignment) * alignment.get_alignment_length()
    return total_gaps / total_length

def calculate_percent_identity(alignment):
    """Calculate the average percent identity across all pairwise comparisons in the alignment."""
    if len(alignment) < 2:
        return 100.0  # Single sequence has 100% identity with itself
    
    total_identity = 0
    num_comparisons = 0
    
    # Compare all pairs of sequences
    for i in range(len(alignment)):
        for j in range(i + 1, len(alignment)):
            seq1 = str(alignment[i].seq)
            seq2 = str(alignment[j].seq)
            
            # Count identical positions (excluding gap-gap pairs)
            identical_positions = 0
            valid_positions = 0
            
            for a, b in zip(seq1, seq2):
                # Skip positions where both sequences have gaps
                if a == "-" and b == "-":
                    continue
                
                valid_positions += 1
                if a == b:
                    identical_positions += 1
            
            # Calculate percent identity for this pair
            if valid_positions > 0:
                pair_identity = (identical_positions / valid_positions) * 100
                total_identity += pair_identity
                num_comparisons += 1
    
    # Return average percent identity across all pairs
    if num_comparisons > 0:
        return total_identity / num_comparisons
    else:
        return 0.0