import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MSABiologicalQualityAnalyzer:
    """
    A specialized analyzer for Multiple Sequence Alignment (MSA) tool performance
    focused on biological quality metrics only. Computational metrics are reported
    for efficiency information but don't affect rankings.
    
    Biological Quality Metrics (used for ranking):
    - BLOSUM Score: Higher is better (better amino acid substitutions)
    - Entropy: Lower is better (more conservation)
    - Gap Fraction: Lower is better (fewer gaps)
    - Percent Identity: Higher is better (more similar sequences)
    
    Efficiency Metrics (reported only):
    - CPU Time: Lower is better (faster)
    - Memory Usage: Lower is better (more efficient)
    """
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        self.data = None
        self.normalized_data = None
        
        # Weights only for biological quality metrics
        self.bio_weights = custom_weights or {
            'blosum_score': 0.3,        # Substitution matrix quality
            'percent_identity': 0.3,    # Sequence similarity
            'entropy': 0.2,             # Conservation measure
            'gap_fraction': 0.2         # Gap management
        }
        
        # Validate weights sum to 1
        if abs(sum(self.bio_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Biological quality weights must sum to 1.0")
    
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
    
    def normalize_biological_metrics(self) -> pd.DataFrame:
        """
        Normalize only biological quality metrics for ranking
        All normalized scores: 0 = worst, 1 = best
        """
        normalized = self.data.copy()
        
        # Biological metrics where HIGHER is BETTER
        higher_better = ['BLOSUM_Score', 'Percent_Identity']
        for col in higher_better:
            if col in normalized.columns:
                col_min, col_max = normalized[col].min(), normalized[col].max()
                if col_max != col_min:  # Avoid division by zero
                    normalized[f'{col}_norm'] = (normalized[col] - col_min) / (col_max - col_min)
                else:
                    normalized[f'{col}_norm'] = 1.0
        
        # Biological metrics where LOWER is BETTER
        lower_better = ['Entropy', 'Gap_Fraction']
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
    
    def calculate_biological_quality_score(self) -> pd.DataFrame:
        """Calculate biological quality score based only on biological metrics"""
        if self.normalized_data is None:
            self.normalize_biological_metrics()
        
        scores = pd.DataFrame(index=self.data.index)
        
        # Calculate weighted biological quality score
        bio_components = []
        
        if 'BLOSUM_Score_norm' in self.normalized_data.columns:
            bio_components.append(self.bio_weights['blosum_score'] * self.normalized_data['BLOSUM_Score_norm'])
        
        if 'Percent_Identity_norm' in self.normalized_data.columns:
            bio_components.append(self.bio_weights['percent_identity'] * self.normalized_data['Percent_Identity_norm'])
            
        if 'Entropy_norm' in self.normalized_data.columns:
            bio_components.append(self.bio_weights['entropy'] * self.normalized_data['Entropy_norm'])
        
        if 'Gap_Fraction_norm' in self.normalized_data.columns:
            bio_components.append(self.bio_weights['gap_fraction'] * self.normalized_data['Gap_Fraction_norm'])
        
        # Sum all biological components
        scores['Biological_Quality_Score'] = sum(bio_components)
        
        # Add individual component scores for analysis
        scores['BLOSUM_Component'] = self.bio_weights['blosum_score'] * self.normalized_data.get('BLOSUM_Score_norm', 0)
        scores['Identity_Component'] = self.bio_weights['percent_identity'] * self.normalized_data.get('Percent_Identity_norm', 0)
        scores['Conservation_Component'] = self.bio_weights['entropy'] * self.normalized_data.get('Entropy_norm', 0)
        scores['Gap_Component'] = self.bio_weights['gap_fraction'] * self.normalized_data.get('Gap_Fraction_norm', 0)
        
        return scores
    
    def rank_tools_by_biology(self) -> pd.DataFrame:
        """Rank tools by biological quality only, in ascending order (1 = best)"""
        scores = self.calculate_biological_quality_score()
        
        rankings = pd.DataFrame(index=self.data.index)
        
        # Calculate biological quality rank (1 = best, higher numbers = worse)
        rankings['Biological_Rank'] = scores['Biological_Quality_Score'].rank(ascending=False, method='dense').astype(int)
        
        # Add biological quality score
        rankings['Biological_Quality_Score'] = scores['Biological_Quality_Score']
        
        # Add original biological metrics
        bio_metrics = ['BLOSUM_Score', 'Entropy', 'Gap_Fraction', 'Percent_Identity']
        for metric in bio_metrics:
            if metric in self.data.columns:
                rankings[metric] = self.data[metric]
        
        # Add efficiency metrics for reporting (not ranking)
        efficiency_metrics = ['CPU_Time', 'Memory_Usage']
        for metric in efficiency_metrics:
            if metric in self.data.columns:
                rankings[metric] = self.data[metric]
        
        # Add component scores for detailed analysis
        component_cols = ['BLOSUM_Component', 'Identity_Component', 'Conservation_Component', 'Gap_Component']
        for col in component_cols:
            rankings[col] = scores[col]
        
        # Sort by biological quality (best first) - ascending rank order
        rankings = rankings.sort_values('Biological_Rank', ascending=True)
        
        return rankings
    
    def generate_biological_explanations(self) -> Dict[str, str]:
        """Generate explanations focused on biological quality performance"""
        rankings = self.rank_tools_by_biology()
        explanations = {}
        
        for tool in rankings.index:
            row = rankings.loc[tool]
            rank = int(row['Biological_Rank'])
            
            strengths = []
            weaknesses = []
            neutral = []
            
            # Analyze biological metrics
            # BLOSUM Score analysis
            blosum_percentile = (rankings['BLOSUM_Score'] >= row['BLOSUM_Score']).mean()
            if blosum_percentile <= 0.3:  # Top 30%
                strengths.append(f"excellent substitution quality (BLOSUM: {row['BLOSUM_Score']:.0f})")
            elif blosum_percentile >= 0.7:  # Bottom 30%
                weaknesses.append(f"suboptimal substitution quality (BLOSUM: {row['BLOSUM_Score']:.0f})")
            else:
                neutral.append(f"moderate substitution quality (BLOSUM: {row['BLOSUM_Score']:.0f})")
            
            # Percent Identity analysis
            if 'Percent_Identity' in row:
                identity_percentile = (rankings['Percent_Identity'] >= row['Percent_Identity']).mean()
                if identity_percentile <= 0.3:  # Top 30%
                    strengths.append(f"high sequence similarity ({row['Percent_Identity']:.1f}% identity)")
                elif identity_percentile >= 0.7:  # Bottom 30%
                    weaknesses.append(f"low sequence similarity ({row['Percent_Identity']:.1f}% identity)")
            
            # Entropy analysis (lower is better for conservation)
            entropy_percentile = (rankings['Entropy'] <= row['Entropy']).mean()
            if entropy_percentile <= 0.3:  # Top 30% (most conserved)
                strengths.append(f"superior conservation (entropy: {row['Entropy']:.4f})")
            elif entropy_percentile >= 0.7:  # Bottom 30% (least conserved)
                weaknesses.append(f"poor conservation (entropy: {row['Entropy']:.4f})")
            else:
                neutral.append(f"moderate conservation (entropy: {row['Entropy']:.4f})")
            
            # Gap management analysis
            gap_percentile = (rankings['Gap_Fraction'] <= row['Gap_Fraction']).mean()
            if gap_percentile <= 0.3:  # Top 30% (fewest gaps)
                strengths.append(f"excellent gap management ({row['Gap_Fraction']:.3f} fraction)")
            elif gap_percentile >= 0.7:  # Bottom 30% (most gaps)
                weaknesses.append(f"excessive gaps ({row['Gap_Fraction']:.3f} fraction)")
            else:
                neutral.append(f"moderate gap management ({row['Gap_Fraction']:.3f} fraction)")
            
            # Build explanation
            explanation = f"**{tool.upper()}** (Biological Rank: #{rank}, Quality Score: {row['Biological_Quality_Score']:.3f})\n"
            
            if strengths:
                explanation += f" **Biological Strengths:** {'; '.join(strengths)}\n"
            if weaknesses:
                explanation += f" **Biological Limitations:** {'; '.join(weaknesses)}\n"
            if neutral:
                explanation += f" **Moderate Performance:** {'; '.join(neutral)}\n"
            
            # Add efficiency information (for reference only)
            if 'CPU_Time' in row and 'Memory_Usage' in row:
                explanation += f" **Computational Efficiency:** {row['CPU_Time']:.2f}s CPU, {row['Memory_Usage']:.1f}MB memory\n"
            
            # Add ranking-based recommendations
            if rank == 1:
                explanation += "**TOP BIOLOGICAL QUALITY** - Best overall biological performance."
            elif rank == 2:
                explanation += " **EXCELLENT BIOLOGICAL QUALITY** - Outstanding biological accuracy."
            elif rank == 3:
                explanation += " **HIGH BIOLOGICAL QUALITY** - Strong biological performance."
            elif rank <= len(rankings) // 2:
                explanation += "**GOOD BIOLOGICAL QUALITY** - Solid biological performance."
            else:
                explanation += "**CONSIDER FOR BIOLOGICAL QUALITY** - May have biological limitations."
            
            explanations[tool] = explanation
        
        return explanations
    
    def create_biological_visualization(self):
        """Create visualization focused on biological quality metrics"""
        scores = self.calculate_biological_quality_score()
        rankings = self.rank_tools_by_biology()
        
        # Set up the plot style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])
        
        # Color palette for consistency
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.data.index)))
        color_dict = dict(zip(self.data.index, colors))
        
        # 1. Biological Quality Ranking (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        scores_sorted = scores.sort_values('Biological_Quality_Score', ascending=True)
        bars = ax1.barh(range(len(scores_sorted)), scores_sorted['Biological_Quality_Score'])
        ax1.set_yticks(range(len(scores_sorted)))
        ax1.set_yticklabels(scores_sorted.index)
        ax1.set_xlabel('Biological Quality Score')
        ax1.set_title('MSA Tools - Biological Quality Ranking', fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # Color bars and add rank labels
        for i, (bar, tool) in enumerate(zip(bars, scores_sorted.index)):
            bar.set_color(color_dict[tool])
            rank = rankings.loc[tool, 'Biological_Rank']
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'Rank #{rank} ({bar.get_width():.3f})', va='center', fontsize=9)
        
        # 2. Biological Components Radar Chart (Top Center)
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        
        components = ['BLOSUM_Component', 'Identity_Component', 'Conservation_Component', 'Gap_Component']
        component_labels = ['BLOSUM Quality', 'Sequence Identity', 'Conservation', 'Gap Management']
        angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for tool in scores.index:
            values = [scores.loc[tool, comp] for comp in components]
            values += values[:1]  # Complete the circle
            ax2.plot(angles, values, 'o-', linewidth=2, label=tool, color=color_dict[tool])
            ax2.fill(angles, values, alpha=0.1, color=color_dict[tool])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(component_labels)
        ax2.set_ylim(0, max([self.bio_weights[k] for k in self.bio_weights.keys()]))
        ax2.set_title('Biological Quality Components', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 3. BLOSUM vs Identity Scatter (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        for tool in self.data.index:
            rank = rankings.loc[tool, 'Biological_Rank']
            ax3.scatter(self.data.loc[tool, 'BLOSUM_Score'], 
                       self.data.loc[tool, 'Percent_Identity'],
                       s=200, color=color_dict[tool], alpha=0.7)
            ax3.annotate(f'{tool} (#{rank})', 
                        (self.data.loc[tool, 'BLOSUM_Score'], 
                         self.data.loc[tool, 'Percent_Identity']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('BLOSUM Score (higher is better)')
        ax3.set_ylabel('Percent Identity (higher is better)')
        ax3.set_title('Substitution Quality vs Sequence Similarity', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Entropy vs Gap Fraction (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        for tool in self.data.index:
            rank = rankings.loc[tool, 'Biological_Rank']
            ax4.scatter(self.data.loc[tool, 'Entropy'], 
                       self.data.loc[tool, 'Gap_Fraction'],
                       s=200, color=color_dict[tool], alpha=0.7)
            ax4.annotate(f'{tool} (#{rank})', 
                        (self.data.loc[tool, 'Entropy'], 
                         self.data.loc[tool, 'Gap_Fraction']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Entropy (lower is better)')
        ax4.set_ylabel('Gap Fraction (lower is better)')
        ax4.set_title('Conservation vs Gap Management', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add ideal quadrant
        ax4.axvline(x=self.data['Entropy'].median(), color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=self.data['Gap_Fraction'].median(), color='red', linestyle='--', alpha=0.5)
        ax4.text(self.data['Entropy'].min(), self.data['Gap_Fraction'].min(), 
                'IDEAL\n(Low entropy,\nFew gaps)', ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        # 5. Individual Metric Rankings (Middle Center)
        ax5 = fig.add_subplot(gs[1, 1])
        
        metric_rankings = pd.DataFrame(index=self.data.index)
        metric_rankings['BLOSUM_Rank'] = self.data['BLOSUM_Score'].rank(ascending=False, method='dense')
        metric_rankings['Identity_Rank'] = self.data['Percent_Identity'].rank(ascending=False, method='dense')
        metric_rankings['Entropy_Rank'] = self.data['Entropy'].rank(ascending=True, method='dense')  # Lower is better
        metric_rankings['Gap_Rank'] = self.data['Gap_Fraction'].rank(ascending=True, method='dense')  # Lower is better
        
        # Create stacked bar chart of ranks
        metrics = ['BLOSUM_Rank', 'Identity_Rank', 'Entropy_Rank', 'Gap_Rank']
        metric_labels = ['BLOSUM', 'Identity', 'Conservation', 'Gaps']
        
        x_pos = np.arange(len(self.data.index))
        bottom = np.zeros(len(self.data.index))
        
        for i, metric in enumerate(metrics):
            ax5.bar(x_pos, metric_rankings[metric], bottom=bottom, 
                   label=metric_labels[i], alpha=0.8)
            bottom += metric_rankings[metric]
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(self.data.index, rotation=45)
        ax5.set_ylabel('Cumulative Rank Score (lower is better)')
        ax5.set_title('Individual Metric Rankings', fontweight='bold')
        ax5.legend()
        
        # 6. Efficiency Information (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Bubble chart: CPU vs Memory, colored by biological rank
        bio_ranks = [rankings.loc[tool, 'Biological_Rank'] for tool in self.data.index]
        scatter = ax6.scatter(self.data['CPU_Time'], self.data['Memory_Usage'],
                            s=200, c=bio_ranks, cmap='RdYlGn_r', alpha=0.7)
        
        for tool in self.data.index:
            rank = rankings.loc[tool, 'Biological_Rank']
            ax6.annotate(f'{tool} (#{rank})', 
                        (self.data.loc[tool, 'CPU_Time'], 
                         self.data.loc[tool, 'Memory_Usage']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax6.set_xlabel('CPU Time (seconds)')
        ax6.set_ylabel('Memory Usage (MB)')
        ax6.set_title('Computational Efficiency\n(color = biological rank)', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax6, label='Biological Rank (1=best)')
        
        # 7. Biological Metrics Heatmap (Bottom)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Prepare biological data for heatmap
        bio_data = self.data[['BLOSUM_Score', 'Entropy', 'Gap_Fraction', 'Percent_Identity']].copy()
        
        # Reorder by biological rank
        bio_data = bio_data.loc[rankings.index]
        
        # Normalize for heatmap display (0-1 scale)
        bio_data_norm = bio_data.copy()
        for col in bio_data_norm.columns:
            if col in ['BLOSUM_Score', 'Percent_Identity']:  # Higher is better
                bio_data_norm[col] = (bio_data_norm[col] - bio_data_norm[col].min()) / \
                                   (bio_data_norm[col].max() - bio_data_norm[col].min())
            else:  # Lower is better (Entropy, Gap_Fraction)
                bio_data_norm[col] = 1 - ((bio_data_norm[col] - bio_data_norm[col].min()) / \
                                        (bio_data_norm[col].max() - bio_data_norm[col].min()))
        
        # Create heatmap
        sns.heatmap(bio_data_norm, annot=bio_data.values, fmt='.3f', 
                   cmap='RdYlGn', center=0.5, 
                   cbar_kws={'label': 'Normalized Score (0=worst, 1=best)'}, ax=ax7)
        ax7.set_title('Biological Quality Metrics (Ordered by Biological Rank)', fontweight='bold')
        ax7.set_xlabel('Biological Metrics')
        ax7.set_ylabel('MSA Tools (in biological quality order)')
        
        # Add rank labels to y-axis
        y_labels = [f"#{int(rankings.loc[tool, 'Biological_Rank'])} {tool}" for tool in rankings.index]
        ax7.set_yticklabels(y_labels, rotation=0)
        
        plt.tight_layout()
        return fig
    
    def generate_biological_recommendations(self) -> Dict[str, str]:
        """Generate recommendations based on biological quality only"""
        rankings = self.rank_tools_by_biology()
        recommendations = {}
        
        # Best overall biological quality
        best_bio = rankings.index[0]  # First in sorted order (rank 1)
        recommendations['Best Biological Quality'] = f"{best_bio} (Rank #1) - Highest overall biological quality (score: {rankings.loc[best_bio, 'Biological_Quality_Score']:.3f})"
        
        # Best specific biological metrics
        best_blosum = rankings.loc[rankings['BLOSUM_Score'].idxmax()].name
        blosum_rank = rankings.loc[best_blosum, 'Biological_Rank']
        recommendations['Best Substitution Quality'] = f"{best_blosum} (Rank #{blosum_rank}) - Highest BLOSUM score ({rankings.loc[best_blosum, 'BLOSUM_Score']:.0f})"
        
        best_identity = rankings.loc[rankings['Percent_Identity'].idxmax()].name
        identity_rank = rankings.loc[best_identity, 'Biological_Rank']
        recommendations['Best Sequence Similarity'] = f"{best_identity} (Rank #{identity_rank}) - Highest percent identity ({rankings.loc[best_identity, 'Percent_Identity']:.1f}%)"
        
        best_conservation = rankings.loc[rankings['Entropy'].idxmin()].name
        conservation_rank = rankings.loc[best_conservation, 'Biological_Rank']
        recommendations['Best Conservation'] = f"{best_conservation} (Rank #{conservation_rank}) - Lowest entropy ({rankings.loc[best_conservation, 'Entropy']:.4f})"
        
        best_gaps = rankings.loc[rankings['Gap_Fraction'].idxmin()].name
        gaps_rank = rankings.loc[best_gaps, 'Biological_Rank']
        recommendations['Best Gap Management'] = f"{best_gaps} (Rank #{gaps_rank}) - Lowest gap fraction ({rankings.loc[best_gaps, 'Gap_Fraction']:.3f})"
        
        # Most efficient (for reference)
        if 'CPU_Time' in rankings.columns and 'Memory_Usage' in rankings.columns:
            fastest_tool = rankings.loc[rankings['CPU_Time'].idxmin()].name
            fastest_rank = rankings.loc[fastest_tool, 'Biological_Rank']
            recommendations['Most Efficient'] = f"{fastest_tool} (Bio Rank #{fastest_rank}) - Fastest execution ({rankings.loc[fastest_tool, 'CPU_Time']:.2f}s, {rankings.loc[fastest_tool, 'Memory_Usage']:.1f}MB)"
        
        return recommendations
    
    def analyze_and_report(self, data_dict: Dict, has_percent_identity: bool = False) -> str:
        """Complete biological quality analysis with comprehensive reporting"""
        
        # Load and analyze data
        self.load_data(data_dict, has_percent_identity)
        rankings = self.rank_tools_by_biology()
        explanations = self.generate_biological_explanations()
        recommendations = self.generate_biological_recommendations()
        
        # Generate comprehensive report
        report = "# MSA Tools - Biological Quality Analysis Report\n\n"
        
        # Executive Summary
        best_tool = rankings.index[0]  # First in biologically ranked order
        best_score = rankings.loc[best_tool, 'Biological_Quality_Score']
        report += f"##  Executive Summary\n\n"
        report += f"** BEST BIOLOGICAL QUALITY: {best_tool.upper()}** (Rank #1, Bio Score: {best_score:.3f})\n\n"
        report += f"Analysis based on biological quality metrics only. Computational efficiency reported for reference.\n"
        report += f"Tools ranked 1 (best) to {len(rankings)} (worst) based purely on biological alignment quality.\n\n"
        
        # Biological Rankings Table
        report += "##  Biological Quality Rankings (Ascending Order)\n\n"
        bio_cols = ['Biological_Rank', 'Biological_Quality_Score', 'BLOSUM_Score', 'Entropy', 'Gap_Fraction', 'Percent_Identity']
        bio_table = rankings[bio_cols].round(4)
        report += "```\n" + bio_table.to_string() + "\n```\n\n"
        
        # Efficiency Reference Table
        if 'CPU_Time' in rankings.columns and 'Memory_Usage' in rankings.columns:
            report += "##  Computational Efficiency Reference\n\n"
            eff_cols = ['Biological_Rank', 'CPU_Time', 'Memory_Usage']
            eff_table = rankings[eff_cols].round(3)
            report += "```\n" + eff_table.to_string() + "\n```\n\n"
        
        # Top performers with detailed analysis
        report += "##  Top Biological Quality Performers\n\n"
        for i in range(min(3, len(rankings))):
            tool = rankings.index[i]
            report += f"### {explanations[tool]}\n\n"
        
        # Specific recommendations
        report += "##  Biological Quality Recommendations\n\n"
        for category, recommendation in recommendations.items():
            report += f"**{category}:** {recommendation}\n\n"
        
        # Complete tool analysis (all tools in biological rank order)
        report += "##  Complete Biological Analysis (By Rank)\n\n"
        for tool in rankings.index:
            report += f"### {explanations[tool]}\n"
        
        # Statistical insights
        report += "##  Biological Quality Statistics\n\n"
        report += f"- **Bio Quality Range:** {rankings['Biological_Quality_Score'].min():.3f} - {rankings['Biological_Quality_Score'].max():.3f}\n"
        report += f"- **BLOSUM Range:** {rankings['BLOSUM_Score'].min():.0f} - {rankings['BLOSUM_Score'].max():.0f}\n"
        report += f"- **Identity Range:** {rankings['Percent_Identity'].min():.1f}% - {rankings['Percent_Identity'].max():.1f}%\n"
        report += f"- **Entropy Range:** {rankings['Entropy'].min():.4f} - {rankings['Entropy'].max():.4f}\n"
        report += f"- **Gap Fraction Range:** {rankings['Gap_Fraction'].min():.3f} - {rankings['Gap_Fraction'].max():.3f}\n"
        
        # Analysis methodology
        report += "\n##  Analysis Methodology\n\n"
        report += "This analysis evaluates MSA tools based on biological quality metrics:\n"
        report += "- **BLOSUM Score:** Measures amino acid substitution quality\n"
        report += "- **Entropy:** Assesses conservation patterns in alignment\n"
        report += "- **Gap Fraction:** Evaluates structural alignment quality\n"
        if has_percent_identity:
            report += "- **Percent Identity:** Direct sequence similarity measurement\n"
        report += "\nComposite biological quality score combines these metrics for overall ranking.\n"
        
        # Footer with timestamp
        from datetime import datetime
        report += f"\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report
