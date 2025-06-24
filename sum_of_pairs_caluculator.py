from Bio import AlignIO
from Bio.Align import substitution_matrices
from itertools import combinations
import argparse
import warnings

def calculate_sp_score_blosum(msa, matrix_name="BLOSUM62") -> int:
    """
    Calculates the Sum-of-Pairs (SP) score for a protein MSA using a substitution matrix.
    
    Args:
        msa (MultipleSeqAlignment): A BioPython MultipleSeqAlignment object.
        matrix_name (str): Name of the substitution matrix. Default is 'BLOSUM62'.

    Returns:
        int: Sum-of-pairs score for the alignment.
    """
    matrix = substitution_matrices.load(matrix_name.upper())
    
    sp_score = 0
    num_seqs = len(msa)

    for col in range(msa.get_alignment_length()):
        column_residues = [record.seq[col] for record in msa]
        for i, j in combinations(range(num_seqs), 2):
            res1 = column_residues[i]
            res2 = column_residues[j]

            if res1 == "-" or res2 == "-":
                continue  # Ignore gaps

            try:
                sp_score += matrix[(res1, res2)]
            except KeyError:
                try:
                    sp_score += matrix[(res2, res1)]
                except KeyError:
                    warnings.warn(f"No substitution score for pair ({res1}, {res2}) in {matrix_name}.")

    return sp_score


def main():
    parser = argparse.ArgumentParser(description="Calculate SP score using BLOSUM62 for an MSA file.")
    parser.add_argument("msa_file", help="Path to the MSA file.")
    parser.add_argument("--format", default="fasta", help="MSA file format (default: fasta). E.g., clustal, phylip")
    args = parser.parse_args()

    try:
        alignment = AlignIO.read(args.msa_file, args.format)
    except Exception as e:
        print(f"Error reading MSA file: {e}")
        return

    score = calculate_sp_score_blosum(alignment)
    print(f"SP score (BLOSUM62): {score}")

if __name__ == "__main__":
    main()

