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
    """Calculate the BLOSUM62 score for the alignment.
    """
    try:
        matrix = substitution_matrices.load("BLOSUM62")
        total_score = 0
        valid_comparisons = 0
        
        # Get all valid matrix keys
        valid_keys = set(matrix.keys())
        
        for i in range(len(alignment)):
            for j in range(i + 1, len(alignment)):
                seq_score = 0
                seq_valid_comparisons = 0
                
                for a, b in zip(str(alignment[i].seq), str(alignment[j].seq)):
                    # Skip gap-gap comparisons
                    if a == "-" and b == "-":
                        continue
                        
                    # Handle gap penalties
                    if a == "-" or b == "-":
                        seq_score -= 4  # Standard gap penalty
                        seq_valid_comparisons += 1
                        continue
                    
                    # Handle standard amino acids
                    if a in valid_keys and b in valid_keys:
                        seq_score += matrix[a][b]
                        seq_valid_comparisons += 1
                    else:
                        # For non-standard amino acids, use a default penalty
                        seq_score -= 1
                        seq_valid_comparisons += 1
                
                if seq_valid_comparisons > 0:
                    total_score += seq_score
                    valid_comparisons += seq_valid_comparisons
        
        # Prevent division by zero
        if valid_comparisons == 0:
            print("Warning: No valid amino acid comparisons found for BLOSUM scoring")
            return 0
            
        # Return the average score
        return int(total_score)
    except Exception as e:
        print(f"Error calculating BLOSUM score: {e}")
        return 0

def calculate_gap_fraction(alignment):
    """Calculate the fraction of gaps in the alignment."""
    total_gaps = sum(str(rec.seq).count("-") for rec in alignment)
    total_length = len(alignment) * alignment.get_alignment_length()
    return total_gaps / total_length
