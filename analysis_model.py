import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MSAPerformanceAnalyzer:
    """
    A comprehensive analyzer for Multiple Sequence Alignment (MSA) tool performance.
    Evaluates tools based on biological accuracy, computational efficiency, and overall quality.
    """
    
    def __init__(self):
        self.data = None
        self.normalized_data = None
        self.weights = {
            'biological_quality': 0.4,  # Entropy + Gap management
            'alignment_accuracy': 0.3,   # BLOSUM score
            'computational_efficiency': 0.3  # CPU + Memory
        }
        
    def load_data(self, data_dict: Dict) -> pd.DataFrame:
        """Load and prepare MSA performance data"""
        self.data = pd.DataFrame(data_dict).T
        self.data.columns = ['BLOSUM_Score', 'Entropy', 'Gap_Fraction', 'CPU_Time', 'Memory_Usage']
        
        # Convert to numeric
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col])
            
        return self.data
    
    def normalize_metrics(self) -> pd.DataFrame:
        """Normalize metrics for fair comparison"""
        normalized = self.data.copy()
        
        # For metrics where higher is better (Entropy)
        normalized['Entropy_norm'] = (normalized['Entropy'] - normalized['Entropy'].min()) / \
                                   (normalized['Entropy'].max() - normalized['Entropy'].min())
        
        # For metrics where lower is better (all others)
        for col in ['BLOSUM_Score', 'Gap_Fraction', 'CPU_Time', 'Memory_Usage']:
            # Invert so higher normalized score = better performance
            normalized[f'{col}_norm'] = 1 - ((normalized[col] - normalized[col].min()) / 
                                           (normalized[col].max() - normalized[col].min()))
        
        self.normalized_data = normalized
        return normalized
    
    def calculate_composite_scores(self) -> pd.DataFrame:
        """Calculate composite performance scores"""
        if self.normalized_data is None:
            self.normalize_metrics()
        
        scores = pd.DataFrame(index=self.data.index)
        
        # Biological Quality Score (Entropy + Gap management)
        scores['Biological_Quality'] = (
            0.6 * self.normalized_data['Entropy_norm'] + 
            0.4 * self.normalized_data['Gap_Fraction_norm']
        )
        
        # Alignment Accuracy Score (BLOSUM)
        scores['Alignment_Accuracy'] = self.normalized_data['BLOSUM_Score_norm']
        
        # Computational Efficiency Score (CPU + Memory)
        scores['Computational_Efficiency'] = (
            0.6 * self.normalized_data['CPU_Time_norm'] + 
            0.4 * self.normalized_data['Memory_Usage_norm']
        )
        
        # Overall Performance Score
        scores['Overall_Performance'] = (
            self.weights['biological_quality'] * scores['Biological_Quality'] +
            self.weights['alignment_accuracy'] * scores['Alignment_Accuracy'] +
            self.weights['computational_efficiency'] * scores['Computational_Efficiency']
        )
        
        return scores
    
    def rank_tools(self) -> pd.DataFrame:
        """Rank tools by different criteria"""
        scores = self.calculate_composite_scores()
        
        rankings = pd.DataFrame(index=self.data.index)
        
        for col in scores.columns:
            rankings[f'{col}_Rank'] = scores[col].rank(ascending=False, method='dense').astype(int)
        
        # Add original metrics for reference
        rankings = pd.concat([rankings, self.data, scores], axis=1)
        
        return rankings.sort_values('Overall_Performance', ascending=False)
    
    def generate_explanations(self) -> Dict[str, str]:
        """Generate detailed explanations for each tool's performance"""
        rankings = self.rank_tools()
        explanations = {}
        
        for tool in rankings.index:
            row = rankings.loc[tool]
            
            # Determine strengths and weaknesses
            strengths = []
            weaknesses = []
            
            # Biological quality analysis
            if row['Biological_Quality'] > 0.7:
                strengths.append(f"excellent biological quality (entropy: {row['Entropy']:.4f})")
            elif row['Biological_Quality'] < 0.3:
                weaknesses.append(f"poor biological quality (entropy: {row['Entropy']:.4f})")
            
            # Gap analysis
            if row['Gap_Fraction'] < 0.25:
                strengths.append(f"low gap fraction ({row['Gap_Fraction']:.4f})")
            elif row['Gap_Fraction'] > 0.35:
                weaknesses.append(f"high gap fraction ({row['Gap_Fraction']:.4f})")
            
            # BLOSUM analysis
            blosum_percentile = (rankings['BLOSUM_Score'] <= row['BLOSUM_Score']).mean()
            if blosum_percentile > 0.7:
                strengths.append(f"strong BLOSUM score ({row['BLOSUM_Score']:.0f})")
            elif blosum_percentile < 0.3:
                weaknesses.append(f"weak BLOSUM score ({row['BLOSUM_Score']:.0f})")
            
            # Computational efficiency
            if row['Computational_Efficiency'] > 0.7:
                strengths.append(f"highly efficient (CPU: {row['CPU_Time']:.2f}s, Memory: {row['Memory_Usage']:.2f}MB)")
            elif row['Computational_Efficiency'] < 0.3:
                weaknesses.append(f"computationally expensive (CPU: {row['CPU_Time']:.2f}s, Memory: {row['Memory_Usage']:.2f}MB)")
            
            # Generate explanation
            explanation = f"**{tool.upper()}** (Rank #{row['Overall_Performance_Rank']}):\n"
            
            if strengths:
                explanation += f"Strengths: {', '.join(strengths)}. "
            if weaknesses:
                explanation += f"Weaknesses: {', '.join(weaknesses)}. "
            
            # Add recommendation
            if row['Overall_Performance_Rank'] == 1:
                explanation += "**RECOMMENDED** - Best overall performance."
            elif row['Overall_Performance_Rank'] <= 3:
                explanation += "Strong performer - good choice for most applications."
            else:
                explanation += "Consider alternatives unless specific strengths match your needs."
            
            explanations[tool] = explanation
        
        return explanations
    
    def create_visualization(self):
        """Create comprehensive visualization of tool performance"""
        scores = self.calculate_composite_scores()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall Performance Ranking
        ax1 = axes[0, 0]
        scores_sorted = scores.sort_values('Overall_Performance', ascending=True)
        bars = ax1.barh(range(len(scores_sorted)), scores_sorted['Overall_Performance'])
        ax1.set_yticks(range(len(scores_sorted)))
        ax1.set_yticklabels(scores_sorted.index)
        ax1.set_xlabel('Overall Performance Score')
        ax1.set_title('Overall MSA Tool Performance')
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            if i >= len(bars) - 2:  # Top 2
                bar.set_color('green')
            elif i >= len(bars) - 4:  # Next 2
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 2. Multi-criteria comparison
        ax2 = axes[0, 1]
        criteria = ['Biological_Quality', 'Alignment_Accuracy', 'Computational_Efficiency']
        x = np.arange(len(scores.index))
        width = 0.25
        
        for i, criterion in enumerate(criteria):
            ax2.bar(x + i*width, scores[criterion], width, label=criterion.replace('_', ' '))
        
        ax2.set_xlabel('Tools')
        ax2.set_ylabel('Normalized Score')
        ax2.set_title('Multi-Criteria Performance Comparison')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(scores.index, rotation=45)
        ax2.legend()
        
        # 3. Entropy vs Gap Fraction (Quality Trade-off)
        ax3 = axes[1, 0]
        scatter = ax3.scatter(self.data['Gap_Fraction'], self.data['Entropy'], 
                            c=scores['Overall_Performance'], cmap='RdYlGn', s=100)
        
        for i, tool in enumerate(self.data.index):
            ax3.annotate(tool, (self.data.loc[tool, 'Gap_Fraction'], 
                              self.data.loc[tool, 'Entropy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Gap Fraction (lower is better)')
        ax3.set_ylabel('Entropy (higher is better)')
        ax3.set_title('Biological Quality Trade-off')
        plt.colorbar(scatter, ax=ax3, label='Overall Performance')
        
        # 4. Efficiency Analysis
        ax4 = axes[1, 1]
        bubble_sizes = 1000 / self.data['Memory_Usage']  # Inverse for bubble size
        scatter2 = ax4.scatter(self.data['CPU_Time'], scores['Overall_Performance'],
                             s=bubble_sizes, c=scores['Computational_Efficiency'], 
                             cmap='RdYlGn', alpha=0.7)
        
        for i, tool in enumerate(self.data.index):
            ax4.annotate(tool, (self.data.loc[tool, 'CPU_Time'], 
                              scores.loc[tool, 'Overall_Performance']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('CPU Time (seconds)')
        ax4.set_ylabel('Overall Performance Score')
        ax4.set_title('Performance vs Efficiency (bubble size = memory efficiency)')
        plt.colorbar(scatter2, ax=ax4, label='Computational Efficiency')
        
        plt.tight_layout()
        return fig
    
    def analyze_and_report(self, data_dict: Dict) -> str:
        """Complete analysis pipeline with automated reporting"""
        # Load and analyze data
        self.load_data(data_dict)
        rankings = self.rank_tools()
        explanations = self.generate_explanations()
        
        # Generate report
        report = "# MSA Tool Performance Analysis Report\n\n"
        
        # Executive Summary
        best_tool = rankings.index[0]
        report += f"## Executive Summary\n"
        report += f"**Recommended Tool: {best_tool.upper()}**\n\n"
        
        # Top 3 recommendations
        report += "## Top 3 Recommendations\n\n"
        for i in range(min(3, len(rankings))):
            tool = rankings.index[i]
            report += f"{i+1}. {explanations[tool]}\n\n"
        
        # Detailed analysis
        report += "## Detailed Performance Metrics\n\n"
        
        # Create summary table
        summary_cols = ['Overall_Performance_Rank', 'BLOSUM_Score', 'Entropy', 
                       'Gap_Fraction', 'CPU_Time', 'Memory_Usage']
        summary_table = rankings[summary_cols].round(4)
        report += summary_table.to_string() + "\n\n"
        
        # Use case recommendations
        report += "## Use Case Recommendations\n\n"
        
        fastest_tool = rankings.loc[rankings['CPU_Time'].idxmin()].name
        most_accurate = rankings.loc[rankings['BLOSUM_Score'].idxmax()].name
        best_quality = rankings.loc[rankings['Biological_Quality'].idxmax()].name
        
        report += f"- **For speed-critical applications**: {fastest_tool}\n"
        report += f"- **For maximum alignment accuracy**: {most_accurate}\n"
        report += f"- **For best biological quality**: {best_quality}\n"
        report += f"- **For balanced performance**: {best_tool}\n\n"
        
        return report

# Example usage and demonstration
if __name__ == "__main__":
    # Your MSA data
    msa_data = {
        'muscle': [-235151, 1.5012, 0.3028, 10.04, 136.08],
        'mafft': [-231210, 1.6107, 0.2470, 0.01, 4.22],
        't_coffee': [-240205, 1.5093, 0.3067, 0.50, 451.25],
        'probcons': [-241003, 1.4247, 0.3462, 7.41, 20.58],
        'kalign': [-233751, 1.6768, 0.2215, 0.10, 2.03],
        'clustalo': [-236404, 1.6462, 0.2424, 1.32, 15.09],
        'prank': [-252833, 1.3172, 0.3936, 11.87, 8.28]
    }
    
    # Initialize analyzer
    analyzer = MSAPerformanceAnalyzer()
    
    # Generate complete analysis report
    report = analyzer.analyze_and_report(msa_data)
    print(report)
    
    # Create visualizations
    fig = analyzer.create_visualization()
    plt.show()
    
    # Get detailed rankings
    rankings = analyzer.rank_tools()
    print("\nDetailed Rankings:")
    print(rankings.to_string())
