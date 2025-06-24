import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MSAPerformanceAnalyzer:
    """
    A comprehensive analyzer for Multiple Sequence Alignment (MSA) tool performance.
    Evaluates tools based on biological accuracy, computational efficiency, and overall quality.
    
    Metric Interpretations:
    - BLOSUM Score: Higher is better (better amino acid substitutions)
    - Entropy: Lower is better (more conservation)
    - Gap Fraction: Lower is better (fewer gaps)
    - Percent Identity: Higher is better (more similar sequences)
    - CPU Time: Lower is better (faster)
    - Memory Usage: Lower is better (more efficient)
    """
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        self.data = None
        self.normalized_data = None
        
        # Default weights - can be customized based on use case
        self.weights = custom_weights or {
            'alignment_quality': 0.6,      # BLOSUM + Percent Identity + Entropy
            'gap_management': 0.2,         # Gap Fraction
            'computational_efficiency': 0.2 # CPU + Memory
        }
        
        # Validate weights sum to 1
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def load_data(self, data_dict: Dict, has_percent_identity: bool = False) -> pd.DataFrame:
        """
        Load and prepare MSA performance data
        
        Parameters:
        -----------
        data_dict : Dict
            Dictionary with tool names as keys and metrics as values
        has_percent_identity : bool
            Whether the data includes percent identity metric
        """
        self.data = pd.DataFrame(data_dict).T
        
        if has_percent_identity:
            self.data.columns = ['BLOSUM_Score', 'Entropy', 'Gap_Fraction', 
                               'Percent_Identity', 'CPU_Time', 'Memory_Usage']
        else:
            self.data.columns = ['BLOSUM_Score', 'Entropy', 'Gap_Fraction', 
                               'CPU_Time', 'Memory_Usage']
            # Estimate percent identity from BLOSUM score if not available
            self.data['Percent_Identity'] = self._estimate_percent_identity()
        
        # Convert to numeric and handle any missing values
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Check for missing values
        if self.data.isnull().any().any():
            print("Warning: Missing values detected. Filling with column medians.")
            self.data = self.data.fillna(self.data.median())
            
        return self.data
    
    def _estimate_percent_identity(self) -> pd.Series:
        """Estimate percent identity from BLOSUM scores (rough approximation)"""
        # Simple linear transformation - this is an approximation
        # In practice, you'd want actual percent identity calculations
        blosum_normalized = (self.data['BLOSUM_Score'] - self.data['BLOSUM_Score'].min()) / \
                           (self.data['BLOSUM_Score'].max() - self.data['BLOSUM_Score'].min())
        return 20 + blosum_normalized * 60  # Scale to reasonable identity range (20-80%)
    
    def normalize_metrics(self) -> pd.DataFrame:
        """
        Normalize metrics for fair comparison using Min-Max normalization
        All normalized scores: 0 = worst, 1 = best
        """
        normalized = self.data.copy()
        
        # Metrics where HIGHER is BETTER
        higher_better = ['BLOSUM_Score', 'Percent_Identity']
        for col in higher_better:
            if col in normalized.columns:
                col_min, col_max = normalized[col].min(), normalized[col].max()
                if col_max != col_min:  # Avoid division by zero
                    normalized[f'{col}_norm'] = (normalized[col] - col_min) / (col_max - col_min)
                else:
                    normalized[f'{col}_norm'] = 1.0
        
        # Metrics where LOWER is BETTER
        lower_better = ['Entropy', 'Gap_Fraction', 'CPU_Time', 'Memory_Usage']
        for col in lower_better:
            if col in normalized.columns:
                col_min, col_max = normalized[col].min(), normalized[col].max()
                if col_max != col_min:  # Avoid division by zero
                    # Invert so higher normalized score = better performance
                    normalized[f'{col}_norm'] = 1 - ((normalized[col] - col_min) / (col_max - col_min))
                else:
                    normalized[f'{col}_norm'] = 1.0
        
        self.normalized_data = normalized
        return normalized
    
    def calculate_composite_scores(self) -> pd.DataFrame:
        """Calculate composite performance scores based on metric categories"""
        if self.normalized_data is None:
            self.normalize_metrics()
        
        scores = pd.DataFrame(index=self.data.index)
        
        # Alignment Quality Score (BLOSUM + Identity + Conservation)
        alignment_components = []
        weights_sum = 0
        
        if 'BLOSUM_Score_norm' in self.normalized_data.columns:
            alignment_components.append(0.4 * self.normalized_data['BLOSUM_Score_norm'])
            weights_sum += 0.4
        
        if 'Percent_Identity_norm' in self.normalized_data.columns:
            alignment_components.append(0.4 * self.normalized_data['Percent_Identity_norm'])
            weights_sum += 0.4
            
        if 'Entropy_norm' in self.normalized_data.columns:
            alignment_components.append(0.2 * self.normalized_data['Entropy_norm'])
            weights_sum += 0.2
        
        if alignment_components:
            scores['Alignment_Quality'] = sum(alignment_components) / weights_sum if weights_sum > 0 else 0
        else:
            scores['Alignment_Quality'] = 0
        
        # Gap Management Score
        scores['Gap_Management'] = self.normalized_data.get('Gap_Fraction_norm', 0)
        
        # Computational Efficiency Score (CPU + Memory)
        efficiency_components = []
        if 'CPU_Time_norm' in self.normalized_data.columns:
            efficiency_components.append(0.6 * self.normalized_data['CPU_Time_norm'])
        if 'Memory_Usage_norm' in self.normalized_data.columns:
            efficiency_components.append(0.4 * self.normalized_data['Memory_Usage_norm'])
        
        if efficiency_components:
            scores['Computational_Efficiency'] = sum(efficiency_components)
        else:
            scores['Computational_Efficiency'] = 0
        
        # Overall Performance Score (weighted combination)
        scores['Overall_Performance'] = (
            self.weights['alignment_quality'] * scores['Alignment_Quality'] +
            self.weights['gap_management'] * scores['Gap_Management'] +
            self.weights['computational_efficiency'] * scores['Computational_Efficiency']
        )
        
        return scores
    
    def rank_tools(self) -> pd.DataFrame:
        """Rank tools by different criteria with detailed metrics"""
        scores = self.calculate_composite_scores()
        
        rankings = pd.DataFrame(index=self.data.index)
        
        # Calculate ranks for each score (1 = best)
        for col in scores.columns:
            rankings[f'{col}_Rank'] = scores[col].rank(ascending=False, method='dense').astype(int)
        
        # Add original metrics and normalized scores
        rankings = pd.concat([rankings, self.data, scores], axis=1)
        
        # Sort by overall performance
        rankings = rankings.sort_values('Overall_Performance', ascending=False)
        
        return rankings
    
    def generate_detailed_explanations(self) -> Dict[str, str]:
        """Generate comprehensive explanations for each tool's performance"""
        rankings = self.rank_tools()
        explanations = {}
        
        for tool in rankings.index:
            row = rankings.loc[tool]
            
            strengths = []
            weaknesses = []
            neutral = []
            
            # Analyze each metric category
            # Alignment Quality
            if row['Alignment_Quality'] > 0.8:
                strengths.append("excellent alignment quality")
            elif row['Alignment_Quality'] > 0.6:
                neutral.append("good alignment quality")
            else:
                weaknesses.append("suboptimal alignment quality")
            
            # BLOSUM Score analysis
            blosum_percentile = (rankings['BLOSUM_Score'] >= row['BLOSUM_Score']).mean()
            if blosum_percentile <= 0.3:  # Top 30%
                strengths.append(f"strong BLOSUM score ({row['BLOSUM_Score']:.0f})")
            elif blosum_percentile >= 0.7:  # Bottom 30%
                weaknesses.append(f"weak BLOSUM score ({row['BLOSUM_Score']:.0f})")
            
            # Entropy analysis (lower is better for conservation)
            entropy_percentile = (rankings['Entropy'] <= row['Entropy']).mean()
            if entropy_percentile <= 0.3:  # Top 30% (lowest entropy)
                strengths.append(f"high conservation (entropy: {row['Entropy']:.4f})")
            elif entropy_percentile >= 0.7:  # Bottom 30% (highest entropy)
                weaknesses.append(f"low conservation (entropy: {row['Entropy']:.4f})")
            
            # Gap management
            if row['Gap_Fraction'] < 0.2:
                strengths.append(f"excellent gap management ({row['Gap_Fraction']:.3f})")
            elif row['Gap_Fraction'] > 0.35:
                weaknesses.append(f"high gap fraction ({row['Gap_Fraction']:.3f})")
            
            # Computational efficiency
            if row['Computational_Efficiency'] > 0.8:
                strengths.append(f"highly efficient (CPU: {row['CPU_Time']:.2f}s, Memory: {row['Memory_Usage']:.1f}MB)")
            elif row['Computational_Efficiency'] < 0.3:
                weaknesses.append(f"computationally expensive (CPU: {row['CPU_Time']:.2f}s, Memory: {row['Memory_Usage']:.1f}MB)")
            
            # Build explanation
            explanation = f"**{tool.upper()}** (Overall Rank: #{row['Overall_Performance_Rank']}, Score: {row['Overall_Performance']:.3f})\n"
            
            if strengths:
                explanation += f"✅ Strengths: {'; '.join(strengths)}\n"
            if weaknesses:
                explanation += f"❌ Weaknesses: {'; '.join(weaknesses)}\n"
            if neutral:
                explanation += f"➖ Neutral: {'; '.join(neutral)}\n"
            
            # Add specific recommendations
            rank = row['Overall_Performance_Rank']
            if rank == 1:
                explanation += "🏆 **TOP CHOICE** - Best overall performance across all metrics."
            elif rank <= 2:
                explanation += "🥈 **EXCELLENT** - Outstanding performance, highly recommended."
            elif rank <= 3:
                explanation += "🥉 **VERY GOOD** - Strong performer for most applications."
            elif rank <= len(rankings) // 2:
                explanation += "✅ **GOOD** - Solid choice with specific strengths."
            else:
                explanation += "⚠️ **CONSIDER ALTERNATIVES** - May have limited use cases."
            
            explanations[tool] = explanation
        
        return explanations
    
    def create_comprehensive_visualization(self):
        """Create detailed visualization suite for MSA tool comparison"""
        scores = self.calculate_composite_scores()
        
        # Set up the plot style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # Color palette for consistency
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.data.index)))
        color_dict = dict(zip(self.data.index, colors))
        
        # 1. Overall Performance Ranking (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        scores_sorted = scores.sort_values('Overall_Performance', ascending=True)
        bars = ax1.barh(range(len(scores_sorted)), scores_sorted['Overall_Performance'])
        ax1.set_yticks(range(len(scores_sorted)))
        ax1.set_yticklabels(scores_sorted.index)
        ax1.set_xlabel('Overall Performance Score')
        ax1.set_title('Overall MSA Tool Performance', fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # Color bars by performance tier
        for i, (bar, tool) in enumerate(zip(bars, scores_sorted.index)):
            bar.set_color(color_dict[tool])
            # Add value labels
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.3f}', va='center', fontsize=9)
        
        # 2. Multi-criteria Radar Chart (Top Center)
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        
        criteria = ['Alignment_Quality', 'Gap_Management', 'Computational_Efficiency']
        angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for tool in scores.index:
            values = [scores.loc[tool, criterion] for criterion in criteria]
            values += values[:1]  # Complete the circle
            ax2.plot(angles, values, 'o-', linewidth=2, label=tool, color=color_dict[tool])
            ax2.fill(angles, values, alpha=0.1, color=color_dict[tool])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([c.replace('_', ' ') for c in criteria])
        ax2.set_ylim(0, 1)
        ax2.set_title('Multi-Criteria Performance', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 3. Quality vs Efficiency Scatter (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        for tool in self.data.index:
            ax3.scatter(scores.loc[tool, 'Computational_Efficiency'], 
                       scores.loc[tool, 'Alignment_Quality'],
                       s=200, color=color_dict[tool], alpha=0.7, label=tool)
            ax3.annotate(tool, 
                        (scores.loc[tool, 'Computational_Efficiency'], 
                         scores.loc[tool, 'Alignment_Quality']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Computational Efficiency')
        ax3.set_ylabel('Alignment Quality')
        ax3.set_title('Quality vs Efficiency Trade-off', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # Add quadrant labels
        ax3.text(0.8, 0.8, 'Ideal\n(Fast & Accurate)', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        ax3.text(0.2, 0.8, 'Accurate\nbut Slow', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        # 4. BLOSUM Score vs Entropy (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        scatter = ax4.scatter(self.data['BLOSUM_Score'], self.data['Entropy'],
                            c=[scores.loc[tool, 'Overall_Performance'] for tool in self.data.index],
                            s=150, cmap='RdYlGn', alpha=0.7)
        
        for tool in self.data.index:
            ax4.annotate(tool, (self.data.loc[tool, 'BLOSUM_Score'], 
                              self.data.loc[tool, 'Entropy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('BLOSUM Score (higher is better)')
        ax4.set_ylabel('Entropy (lower is better)')
        ax4.set_title('Alignment Accuracy vs Conservation', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Overall Performance')
        
        # 5. Resource Usage Analysis (Middle Center)
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Create bubble chart: CPU vs Memory, bubble size = overall performance
        bubble_sizes = [scores.loc[tool, 'Overall_Performance'] * 1000 for tool in self.data.index]
        
        for i, tool in enumerate(self.data.index):
            ax5.scatter(self.data.loc[tool, 'CPU_Time'], self.data.loc[tool, 'Memory_Usage'],
                       s=bubble_sizes[i], color=color_dict[tool], alpha=0.6, label=tool)
            ax5.annotate(tool, (self.data.loc[tool, 'CPU_Time'], 
                              self.data.loc[tool, 'Memory_Usage']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax5.set_xlabel('CPU Time (seconds, lower is better)')
        ax5.set_ylabel('Memory Usage (MB, lower is better)')
        ax5.set_title('Resource Usage\n(bubble size = overall performance)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Gap Analysis (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        gap_data = self.data['Gap_Fraction'].sort_values()
        bars = ax6.bar(range(len(gap_data)), gap_data.values)
        ax6.set_xticks(range(len(gap_data)))
        ax6.set_xticklabels(gap_data.index, rotation=45)
        ax6.set_ylabel('Gap Fraction')
        ax6.set_title('Gap Management Comparison', fontweight='bold')
        
        for i, (bar, tool) in enumerate(zip(bars, gap_data.index)):
            bar.set_color(color_dict[tool])
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        
        # 7. Performance Metrics Heatmap (Bottom)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Prepare data for heatmap
        heatmap_data = self.normalized_data[[col for col in self.normalized_data.columns 
                                           if col.endswith('_norm')]].copy()
        heatmap_data.columns = [col.replace('_norm', '').replace('_', ' ') for col in heatmap_data.columns]
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5, 
                   fmt='.3f', cbar_kws={'label': 'Normalized Score (0=worst, 1=best)'}, ax=ax7)
        ax7.set_title('Normalized Performance Metrics Heatmap', fontweight='bold')
        ax7.set_xlabel('Metrics')
        ax7.set_ylabel('Tools')
        
        plt.tight_layout()
        return fig
    
    def generate_use_case_recommendations(self) -> Dict[str, str]:
        """Generate specific recommendations for different use cases"""
        rankings = self.rank_tools()
        recommendations = {}
        
        # Speed-critical applications
        fastest_tool = rankings.loc[rankings['CPU_Time'].idxmin()].name
        recommendations['Speed Critical'] = f"{fastest_tool} - Fastest execution time ({rankings.loc[fastest_tool, 'CPU_Time']:.2f}s)"
        
        # Memory-constrained environments
        memory_efficient = rankings.loc[rankings['Memory_Usage'].idxmin()].name
        recommendations['Memory Constrained'] = f"{memory_efficient} - Lowest memory usage ({rankings.loc[memory_efficient, 'Memory_Usage']:.1f}MB)"
        
        # Highest alignment quality
        best_quality = rankings.loc[rankings['Alignment_Quality'].idxmax()].name
        recommendations['Quality Focused'] = f"{best_quality} - Best alignment quality (score: {rankings.loc[best_quality, 'Alignment_Quality']:.3f})"
        
        # Best gap management
        best_gaps = rankings.loc[rankings['Gap_Fraction'].idxmin()].name
        recommendations['Minimal Gaps'] = f"{best_gaps} - Best gap management ({rankings.loc[best_gaps, 'Gap_Fraction']:.3f} gap fraction)"
        
        # Balanced performance
        best_overall = rankings.index[0]
        recommendations['Balanced Use'] = f"{best_overall} - Best overall performance (score: {rankings.loc[best_overall, 'Overall_Performance']:.3f})"
        
        # Conservation-focused
        best_conservation = rankings.loc[rankings['Entropy'].idxmin()].name
        recommendations['Conservation Focused'] = f"{best_conservation} - Highest conservation (entropy: {rankings.loc[best_conservation, 'Entropy']:.4f})"
        
        return recommendations
    
    def analyze_and_report(self, data_dict: Dict, has_percent_identity: bool = False) -> str:
        """Complete analysis pipeline with comprehensive reporting"""
        # Load and analyze data
        self.load_data(data_dict, has_percent_identity)
        rankings = self.rank_tools()
        explanations = self.generate_detailed_explanations()
        use_cases = self.generate_use_case_recommendations()
        
        # Generate comprehensive report
        report = "# 🧬 MSA Tool Performance Analysis Report\n\n"
        
        # Executive Summary
        best_tool = rankings.index[0]
        best_score = rankings.loc[best_tool, 'Overall_Performance']
        report += f"## 📊 Executive Summary\n\n"
        report += f"**🏆 RECOMMENDED TOOL: {best_tool.upper()}** (Overall Score: {best_score:.3f})\n\n"
        report += f"Based on comprehensive analysis of {len(rankings)} MSA tools across multiple performance criteria.\n\n"
        
        # Top 3 recommendations with detailed analysis
        report += "## 🥇 Top 3 Performance Rankings\n\n"
        for i in range(min(3, len(rankings))):
            tool = rankings.index[i]
            report += f"### {i+1}. {explanations[tool]}\n\n"
        
        # Performance summary table
        report += "## 📈 Performance Metrics Summary\n\n"
        summary_cols = ['Overall_Performance_Rank', 'Overall_Performance', 
                       'BLOSUM_Score', 'Entropy', 'Gap_Fraction', 
                       'CPU_Time', 'Memory_Usage']
        
        if 'Percent_Identity' in rankings.columns:
            summary_cols.insert(-2, 'Percent_Identity')
        
        summary_table = rankings[summary_cols].round(4)
        report += "```\n" + summary_table.to_string() + "\n```\n\n"
        
        # Use case specific recommendations
        report += "## 🎯 Use Case Specific Recommendations\n\n"
        for use_case, recommendation in use_cases.items():
            report += f"**{use_case}:** {recommendation}\n\n"
        
        # Detailed tool analysis
        report += "## 🔍 Detailed Tool Analysis\n\n"
        for tool in rankings.index:
            report += f"### {explanations[tool]}\n"
        
        # Statistical insights
        report += "## 📊 Statistical Insights\n\n"
        report += f"- **Performance Range:** {rankings['Overall_Performance'].min():.3f} - {rankings['Overall_Performance'].max():.3f}\n"
        report += f"- **Speed Range:** {rankings['CPU_Time'].min():.2f}s - {rankings['CPU_Time'].max():.2f}s\n"
        report += f"- **Memory Range:** {rankings['Memory_Usage'].min():.1f}MB - {rankings['Memory_Usage'].max():.1f}MB\n"
        report += f"- **Gap Fraction Range:** {rankings['Gap_Fraction'].min():.3f} - {rankings['Gap_Fraction'].max():.3f}\n\n"
        
        # Methodology note
        report += "## ⚙️ Analysis Methodology\n\n"
        report += f"**Weighting Scheme:**\n"
        for category, weight in self.weights.items():
            report += f"- {category.replace('_', ' ').title()}: {weight:.1%}\n"
        report += "\n**Normalization:** Min-Max scaling (0=worst, 1=best) applied to all metrics.\n"
        report += "**Scoring:** Composite scores calculated as weighted averages of normalized metrics.\n\n"
        
        return report

# Example usage with real MSA performance data
if __name__ == "__main__":
    # MSA performance data with percent identity included
    # Format: [BLOSUM_Score, Entropy, Gap_Fraction, Percent_Identity, CPU_Time, Memory_Usage]
    msa_data = {
        'muscle': [-795594, 0.7353, 0.7520, 25.32, 34.46, 124.29],
        't_coffee': [-801992, 0.7613, 0.7434, 25.12, 1.85, 485.16],
        'kalign': [-800662, 1.1459, 0.6171, 24.60, 0.10, 1.72],
        'prank': [-1152181, 0.7109, 0.7857, 21.99, 97.03, 13.11],
        'mafft': [-778458, 1.0342, 0.6456, 25.43, 0.01, 4.22],
        'probcons': [-789476, 0.8281, 0.7203, 25.25, 15.93, 32.43],
        'clustalo': [-830552, 1.0767, 0.6428, 24.50, 7.67, 17.52]
    }
    
    # Initialize analyzer with custom weights (optional)
    custom_weights = {
        'alignment_quality': 0.5,      # Emphasize alignment quality
        'gap_management': 0.2,         # Moderate gap importance
        'computational_efficiency': 0.3 # Balance efficiency needs
    }
    
    analyzer = MSAPerformanceAnalyzer(custom_weights)
    
    # Generate complete analysis report (now with percent identity data)
    print("Generating comprehensive MSA analysis report...")
    report = analyzer.analyze_and_report(msa_data, has_percent_identity=True)
    print(report)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig = analyzer.create_comprehensive_visualization()
    plt.show()
    
    # Optional: Save detailed rankings to CSV
    rankings = analyzer.rank_tools()
    # rankings.to_csv('msa_tool_rankings.csv', index=True)
    print("\nAnalysis complete! Check the generated visualizations and report above.")