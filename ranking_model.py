import json
from typing import Dict, List
from dataclasses import dataclass
import uuid

@dataclass
class MSAResult:
    tool: str
    blosum_score: float
    entropy: float
    gap_fraction: float
    cpu_time_sec: float
    memory_usage_mb: float

class MSAnalyzer:
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights for scoring (can be adjusted)
        self.weights = weights or {
            "blosum_score": 0.5,  # Primary metric
            "entropy": 0.3,       # Secondary metric
            "gap_fraction": 0.2   # Tertiary metric
        }

    def load_data(self, data: Dict) -> List[MSAResult]:
        """Parse input data into MSAResult objects."""
        results = []
        for item in data["results"]:
            results.append(
                MSAResult(
                    tool=item["tool"],
                    blosum_score=item["blosum_score"],
                    entropy=item["entropy"],
                    gap_fraction=item["gap_fraction"],
                    cpu_time_sec=item["cpu_time_sec"],
                    memory_usage_mb=item["memory_usage_mb"]
                )
            )
        return results

    def normalize_metric(self, values: List[float], reverse: bool = False) -> List[float]:
        """Normalize metrics to [0, 1]. If reverse=True, lower is better."""
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return [1.0] * len(values)
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
        return [1.0 - n if reverse else n for n in normalized]

    def compute_scores(self, results: List[MSAResult]) -> List[Dict]:
        """Compute weighted scores for each tool."""
        blosum_scores = [r.blosum_score for r in results]
        entropies = [r.entropy for r in results]
        gap_fractions = [r.gap_fraction for r in results]

        # Normalize metrics (BLOSUM: higher better, entropy/gap: lower better)
        norm_blosum = self.normalize_metric(blosum_scores, reverse=False)
        norm_entropy = self.normalize_metric(entropies, reverse=True)
        norm_gap = self.normalize_metric(gap_fractions, reverse=True)

        scores = []
        for i, result in enumerate(results):
            score = (
                self.weights["blosum_score"] * norm_blosum[i]
                + self.weights["entropy"] * norm_entropy[i]
                + self.weights["gap_fraction"] * norm_gap[i]
            )
            scores.append({
                "tool": result.tool,
                "score": score,
                "blosum_score": result.blosum_score,
                "entropy": result.entropy,
                "gap_fraction": result.gap_fraction,
                "cpu_time_sec": result.cpu_time_sec,
                "memory_usage_mb": result.memory_usage_mb
            })
        return sorted(scores, key=lambda x: x["score"], reverse=True)

    def generate_report(self, scores: List[Dict]) -> str:
        """Generate a markdown report with analysis and recommendation."""
        best_tool = scores[0]
        report = f"# MSA Analysis Report\n\n**Session ID**: {uuid.uuid4()}\n\n"
        report += "## Summary\nThe following MSA tools were evaluated based on BLOSUM score (higher is better), entropy (lower is better), and gap fraction (lower is better). Computational efficiency (CPU time and memory usage) was considered as a secondary factor.\n\n"
        
        report += "## Results Table\n\n| Tool | BLOSUM Score | Entropy | Gap Fraction | CPU Time (sec) | Memory Usage (MB) | Score |\n"
        report += "|------|--------------|---------|--------------|----------------|-------------------|-------|\n"
        for s in scores:
            report += f"| {s['tool']} | {s['blosum_score']:,} | {s['entropy']:.4f} | {s['gap_fraction']:.4f} | {s['cpu_time_sec']:.2f} | {s['memory_usage_mb']:.2f} | {s['score']:.4f} |\n"
        
        report += f"\n## Recommendation\n**Best Tool**: {best_tool['tool'].capitalize()}\n\n### Explanation\n"
        report += f"- **BLOSUM Score**: {best_tool['tool'].capitalize()} achieved a BLOSUM score of {best_tool['blosum_score']:,}, indicating {'the best' if best_tool['blosum_score'] == max(s['blosum_score'] for s in scores) else 'a competitive'} residue matching quality.\n"
        report += f"- **Entropy**: With an entropy of {best_tool['entropy']:.4f}, it {'has the most conserved alignment' if best_tool['entropy'] == min(s['entropy'] for s in scores) else 'shows reasonable conservation'}.\n"
        report += f"- **Gap Fraction**: A gap fraction of {best_tool['gap_fraction']:.4f} suggests {'the most compact alignment' if best_tool['gap_fraction'] == min(s['gap_fraction'] for s in scores) else 'a moderately compact alignment'}.\n"
        report += f"- **Computational Efficiency**: It used {best_tool['cpu_time_sec']:.2f} seconds and {best_tool['memory_usage_mb']:.2f} MB, which is {'highly efficient' if best_tool['cpu_time_sec'] == min(s['cpu_time_sec'] for s in scores) else 'acceptable but not the fastest'}.\n\n"

        report += "### Trade-Offs\n"
        for s in scores[1:]:
            if abs(s["score"] - best_tool["score"]) < 0.1:  # Close competitors
                report += f"- **{s['tool'].capitalize()}**: Offers a BLOSUM score of {s['blosum_score']:,}, entropy of {s['entropy']:.4f}, and gap fraction of {s['gap_fraction']:.4f}. "
                report += f"It is {'faster' if s['cpu_time_sec'] < best_tool['cpu_time_sec'] else 'slower'} ({s['cpu_time_sec']:.2f} sec) and uses {'less' if s['memory_usage_mb'] < best_tool['memory_usage_mb'] else 'more'} memory ({s['memory_usage_mb']:.2f} MB). "
                report += f"Consider {s['tool']} if {'speed' if s['cpu_time_sec'] < best_tool['cpu_time_sec'] else 'alignment conservation' if s['entropy'] < best_tool['entropy'] else 'fewer gaps' if s['gap_fraction'] < best_tool['gap_fraction'] else 'other factors'} are prioritized.\n"

        report += "\n### Conclusion\n"
        report += f"{best_tool['tool'].capitalize()} is recommended for its {'superior BLOSUM score' if best_tool['blosum_score'] == max(s['blosum_score'] for s in scores) else 'balanced performance'} and overall alignment quality. "
        report += "Adjust the weights in the model if specific metrics (e.g., gap fraction or computational efficiency) are more critical for your use case."
        
        return report

    def analyze(self, data: Dict) -> str:
        """Main method to analyze MSA data and return a report."""
        results = self.load_data(data)
        scores = self.compute_scores(results)
        return self.generate_report(scores)

# Example usage
if __name__ == "__main__":
    data = {
        "session_id": "0cf8b332-a4f8-40ab-8e93-132cb806397e",
        "results": [
            {"tool": "muscle", "blosum_score": -1362295, "entropy": 1.3455008900558034, "gap_fraction": 0.3946050096339114, "cpu_time_sec": 103.16, "memory_usage_mb": 420.66015625},
            {"tool": "mafft", "blosum_score": -1364402, "entropy": 1.486567627107301, "gap_fraction": 0.3300639658848614, "cpu_time_sec": 0.01, "memory_usage_mb": 4.21875},
            {"tool": "t_coffee", "blosum_score": -1386585, "entropy": 1.37416379619616, "gap_fraction": 0.3857282502443793, "cpu_time_sec": 2.61, "memory_usage_mb": 502.34375},
            {"tool": "probcons", "blosum_score": -1370478, "entropy": 1.3095481502821744, "gap_fraction": 0.41325863678804853, "cpu_time_sec": 85.38, "memory_usage_mb": 50.01953125},
            {"tool": "kalign", "blosum_score": -1384737, "entropy": 1.4744551947036326, "gap_fraction": 0.3460978147762747, "cpu_time_sec": 0.18179519176483155, "memory_usage_mb": 3.4375},
            {"tool": "clustalo", "blosum_score": -1393753, "entropy": 1.4998217695101115, "gap_fraction": 0.33432203389830506, "cpu_time_sec": 26.6, "memory_usage_mb": 35.8984375},
            {"tool": "prank", "blosum_score": -1550175, "entropy": 1.1009310948442945, "gap_fraction": 0.5257358490566038, "cpu_time_sec": 76.98, "memory_usage_mb": 16.828125}
        ]
    }
    
    analyzer = MSAnalyzer()
    report = analyzer.analyze(data)
    print(report)
