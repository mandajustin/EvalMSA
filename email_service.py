import smtplib
import os
import csv
import tempfile
import base64
import io
import shutil
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Import the MSA biological quality analyzer from the second file
from msa_analyzer import MSABiologicalQualityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEmailService:
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587, 
                 sender_email: str = None, sender_password: str = None):
        """
        Initialize enhanced email service with SMTP configuration and MSA biological analysis
        
        Args:
            smtp_server: SMTP server address (default: Gmail)
            smtp_port: SMTP server port (default: 587 for TLS)
            sender_email: Sender's email address
            sender_password: Sender's email password or app password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = "justinmandah@gmail.com" #sender_email or os.getenv('SENDER_EMAIL')
        self.sender_password = "zxzj vwgi guwp euab"#sender_password or os.getenv('SENDER_PASSWORD')
        #self.sender_email = sender_email or os.getenv('SENDER_EMAIL')
        #self.sender_password = sender_password or os.getenv('SENDER_PASSWORD')
        
        if not self.sender_email or not self.sender_password:
            raise ValueError("Sender email and password must be provided via parameters or environment variables")
        
        # Initialize MSA biological quality analyzer with balanced biological weights
        self.msa_analyzer = MSABiologicalQualityAnalyzer({
            'blosum_score': 0.3,        # Substitution matrix quality
            'percent_identity': 0.3,    # Sequence similarity
            'entropy': 0.2,             # Conservation measure
            'gap_fraction': 0.2         # Gap management
        })

    def backup_attachments(self, session_id: str, csv_file_path: str = None, 
                          report_file_path: str = None, chart_file_path: str = None, 
                          alignment_files: Dict[str, str] = None):
        """
        Backup attachment files to home folder
        
        Args:
            session_id: Session identifier
            csv_file_path: Path to CSV file
            report_file_path: Path to report file
            chart_file_path: Path to chart file
            alignment_files: Dictionary of alignment files
        """
        try:
            # Create backup directory in home folder
            backup_dir = Path.home() / "msa_backups" / f"session_{session_id[:8]}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup files if they exist
            if csv_file_path and os.path.exists(csv_file_path):
                shutil.copy2(csv_file_path, backup_dir / f"msa_results_{session_id[:8]}.csv")
            
            if report_file_path and os.path.exists(report_file_path):
                shutil.copy2(report_file_path, backup_dir / f"msa_biological_analysis_{session_id[:8]}.md")
            
            if chart_file_path and os.path.exists(chart_file_path):
                shutil.copy2(chart_file_path, backup_dir / f"msa_biological_chart_{session_id[:8]}.png")
            
            if alignment_files:
                for tool_name, file_path in alignment_files.items():
                    if os.path.exists(file_path):
                        shutil.copy2(file_path, backup_dir / f"{tool_name}_alignment_{session_id[:8]}.fasta")
            
            logger.info(f"Files backed up to: {backup_dir}")
            
        except Exception as e:
            logger.warning(f"Backup failed: {str(e)}")

    def create_results_csv(self, results: List[Dict[str, Any]], session_id: str) -> str:
        """
        Create a CSV file from evaluation results
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            
        Returns:
            Path to the created CSV file
        """
        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_results_{session_id}.csv', 
                                               delete=False, newline='')
        
        if not results:
            logger.warning("No results to write to CSV")
            return csv_file.name
            
        fieldnames = ['tool', 'blosum_score', 'entropy', 'gap_fraction', 'percent_identity', 'cpu_time_sec', 'memory_usage_mb']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            if hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                result_dict = result
            writer.writerow(result_dict)
        
        csv_file.close()
        logger.info(f"CSV file created: {csv_file.name}")
        return csv_file.name

    def create_biological_analysis_report(self, results: List[Dict[str, Any]], session_id: str) -> str:
        """
        Create comprehensive MSA biological quality analysis report
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            
        Returns:
            Path to the created report file
        """
        try:
            # Convert results to the format expected by MSA biological analyzer
            msa_data = {}
            for result in results:
                if hasattr(result, 'dict'):
                    r = result.dict()
                else:
                    r = result
                
                # Format: [BLOSUM_Score, Entropy, Gap_Fraction, Percent_Identity, CPU_Time, Memory_Usage]
                msa_data[r['tool']] = [
                    r['blosum_score'],
                    r['entropy'],
                    r['gap_fraction'],
                    r.get('percent_identity', 0.0),
                    r['cpu_time_sec'],
                    r['memory_usage_mb']
                ]
            
            # Generate comprehensive biological analysis report
            report_content = self.msa_analyzer.analyze_and_report(msa_data, has_percent_identity=True)
            
            # Save report to temporary file
            report_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_biological_analysis_{session_id}.md', 
                                                      delete=False, encoding='utf-8')
            report_file.write(report_content)
            report_file.close()
            
            logger.info(f"Biological analysis report created: {report_file.name}")
            return report_file.name
            
        except Exception as e:
            logger.error(f"Failed to create biological analysis report: {str(e)}")
            return None

    def create_biological_visualization_chart(self, results: List[Dict[str, Any]], session_id: str) -> str:
        """
        Create comprehensive biological quality visualization chart
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            
        Returns:
            Path to the created chart file
        """
        try:
            # Convert results to the format expected by MSA biological analyzer
            msa_data = {}
            for result in results:
                if hasattr(result, 'dict'):
                    r = result.dict()
                else:
                    r = result
                
                msa_data[r['tool']] = [
                    r['blosum_score'],
                    r['entropy'],
                    r['gap_fraction'],
                    r.get('percent_identity', 0.0),
                    r['cpu_time_sec'],
                    r['memory_usage_mb']
                ]
            
            # Load data and create biological visualization
            self.msa_analyzer.load_data(msa_data, has_percent_identity=True)
            fig = self.msa_analyzer.create_biological_visualization()
            
            # Save chart to temporary file
            chart_file = tempfile.NamedTemporaryFile(suffix=f'_biological_chart_{session_id}.png', 
                                                     delete=False)
            fig.savefig(chart_file.name, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Free up memory
            
            logger.info(f"Biological visualization chart created: {chart_file.name}")
            return chart_file.name
            
        except Exception as e:
            logger.error(f"Failed to create biological visualization chart: {str(e)}")
            return None

    def get_biological_analysis_summary(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Get a biological quality analysis summary for email body
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with biological analysis summary
        """
        try:
            # Convert results to MSA biological analyzer format
            msa_data = {}
            for result in results:
                if hasattr(result, 'dict'):
                    r = result.dict()
                else:
                    r = result
                
                msa_data[r['tool']] = [
                    r['blosum_score'],
                    r['entropy'],
                    r['gap_fraction'],
                    r.get('percent_identity', 0.0),
                    r['cpu_time_sec'],
                    r['memory_usage_mb']
                ]
            
            # Load data and get biological rankings and recommendations
            self.msa_analyzer.load_data(msa_data, has_percent_identity=True)
            rankings = self.msa_analyzer.rank_tools_by_biology()
            recommendations = self.msa_analyzer.generate_biological_recommendations()
            
            # Get top biological performer
            best_tool = rankings.index[0]  # First in biologically ranked order
            best_score = rankings.loc[best_tool, 'Biological_Quality_Score']
            best_rank = rankings.loc[best_tool, 'Biological_Rank']
            
            # Get specific metric leaders
            best_blosum = rankings.loc[rankings['BLOSUM_Score'].idxmax()].name
            best_conservation = rankings.loc[rankings['Entropy'].idxmin()].name
            best_gaps = rankings.loc[rankings['Gap_Fraction'].idxmin()].name
            
            return {
                'best_biological_tool': best_tool,
                'best_biological_score': f"{best_score:.3f}",
                'best_biological_rank': str(best_rank),
                'total_tools': len(rankings),
                'recommendations': recommendations,
                'best_blosum': best_blosum,
                'best_conservation': best_conservation,
                'best_gaps': best_gaps,
                'blosum_score': f"{rankings.loc[best_blosum, 'BLOSUM_Score']:.0f}",
                'entropy_score': f"{rankings.loc[best_conservation, 'Entropy']:.4f}",
                'gap_score': f"{rankings.loc[best_gaps, 'Gap_Fraction']:.3f}"
            }
            
        except Exception as e:
            logger.error(f"Failed to get biological analysis summary: {str(e)}")
            return {
                'best_biological_tool': 'N/A',
                'best_biological_score': 'N/A',
                'best_biological_rank': 'N/A',
                'total_tools': len(results),
                'recommendations': {},
                'best_blosum': 'N/A',
                'best_conservation': 'N/A',
                'best_gaps': 'N/A',
                'blosum_score': 'N/A',
                'entropy_score': 'N/A',
                'gap_score': 'N/A'
            }

    def send_evaluation_results(self, recipient_email: str, results: List[Dict[str, Any]], 
                                alignment_files: Dict[str, str], session_id: str, 
                                sequence_info: str = "", include_analysis: bool = True) -> bool:
        """
        Send evaluation results with comprehensive biological quality analysis via email,
        and always back up files to the home directory regardless of success or failure.
        """
        csv_file_path = None
        report_file_path = None
        chart_file_path = None

        try:
            # Create message container
            msg = MIMEMultipart('mixed')
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"MSA Biological Quality Analysis Results - Session {session_id[:8]}"

            # Get biological analysis summary for email body
            analysis_summary = self.get_biological_analysis_summary(results) if include_analysis else {}

            # Create email body
            body = self._create_enhanced_email_body(results, session_id, sequence_info, analysis_summary)
            msg.attach(MIMEText(body, 'html'))

            # Create and attach CSV file with results
            csv_file_path = self.create_results_csv(results, session_id)
            self._attach_file(msg, csv_file_path, f"msa_results_{session_id[:8]}.csv")

            # Create and attach comprehensive biological analysis report
            if include_analysis and len(results) > 1:
                report_file_path = self.create_biological_analysis_report(results, session_id)
                if report_file_path:
                    self._attach_file(msg, report_file_path, f"msa_biological_analysis_{session_id[:8]}.md")

                chart_file_path = self.create_biological_visualization_chart(results, session_id)
                if chart_file_path:
                    self._attach_file(msg, chart_file_path, f"msa_biological_chart_{session_id[:8]}.png")

            # Attach alignment files
            for tool_name, file_path in alignment_files.items():
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    filename = f"{tool_name}_alignment_{session_id[:8]}.fasta"
                    self._attach_file(msg, file_path, filename)
                else:
                    logger.warning(f"Alignment file for {tool_name} not found or empty: {file_path}")

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Enhanced biological quality email sent successfully to {recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send enhanced biological quality email: {str(e)}")
            return False

        finally:
            # Backup attachments even if email sending fails
            self.backup_attachments(
                session_id=session_id,
                csv_file_path=csv_file_path,
                report_file_path=report_file_path,
                chart_file_path=chart_file_path,
                alignment_files=alignment_files
            )

            # Clean up temporary files
            try:
                if csv_file_path and os.path.exists(csv_file_path):
                    os.unlink(csv_file_path)
                if include_analysis:
                    if report_file_path and os.path.exists(report_file_path):
                        os.unlink(report_file_path)
                    if chart_file_path and os.path.exists(chart_file_path):
                        os.unlink(chart_file_path)
            except OSError as e:
                logger.warning(f"Could not delete temporary files: {str(e)}")

    def _attach_file(self, msg: MIMEMultipart, file_path: str, filename: str):
        """
        Attach a file to the email message
        
        Args:
            msg: Email message object
            file_path: Path to the file to attach
            filename: Name for the attached file
        """
        try:
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            msg.attach(part)
            logger.info(f"File attached: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to attach file {file_path}: {str(e)}")

    def _create_enhanced_email_body(self, results: List[Dict[str, Any]], session_id: str, 
                                  sequence_info: str, analysis_summary: Dict[str, str]) -> str:
        """
        Create enhanced HTML email body with biological quality analysis summary
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            sequence_info: Information about the sequences
            analysis_summary: Biological analysis summary from MSA analyzer
            
        Returns:
            HTML formatted email body
        """
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .biological-highlight {{ 
                    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
                    color: white; padding: 15px; border-radius: 8px; margin: 15px 0;
                    text-align: center; font-size: 18px; font-weight: bold;
                }}
                .metric-leaders {{ 
                    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                    color: #333; padding: 15px; border-radius: 8px; margin: 15px 0;
                }}
                .results-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .results-table th, .results-table td {{ 
                    border: 1px solid #ddd; padding: 10px; text-align: center; 
                }}
                .results-table th {{ 
                    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                    color: white; font-weight: bold;
                }}
                .results-table tr:nth-child(even) {{ background-color: #f8fff8; }}
                .results-table tr:hover {{ background-color: #e8f5e8; }}
                .recommendations {{ 
                    background-color: #f0fff0; padding: 15px; border-radius: 8px; 
                    border-left: 5px solid #56ab2f; margin: 20px 0;
                }}
                .metric-explanation {{ 
                    background-color: #f9f9f9; padding: 15px; border-radius: 8px;
                    border-left: 4px solid #2196F3; margin: 15px 0; 
                }}
                .attachments {{ 
                    background-color: #fff3cd; padding: 15px; border-radius: 8px;
                    border-left: 4px solid #ffc107; margin: 20px 0;
                }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; 
                          border-top: 1px solid #eee; padding-top: 15px; }}
                .best-bio-tool {{ color: #28a745; font-weight: bold; font-size: 1.1em; }}
                .biological-focus {{ 
                    background-color: #e8f5e8; padding: 10px; border-radius: 5px;
                    border-left: 3px solid #56ab2f; margin: 10px 0;
                }}
                ul {{ padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🧬 Multiple Sequence Alignment - Biological Quality Analysis</h2>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                {f'<p><strong>Sequence Info:</strong> {sequence_info}</p>' if sequence_info else ''}
            </div>
        """
        
        # Add biological quality highlight if available
        if analysis_summary.get('best_biological_tool') != 'N/A':
            html_body += f"""
            <div class="biological-highlight">
                🏆 BEST BIOLOGICAL QUALITY: {analysis_summary['best_biological_tool'].upper()} 
                (Bio Score: {analysis_summary['best_biological_score']}, Rank #{analysis_summary['best_biological_rank']})
            </div>
            """
        
        # Add metric leaders section
        if analysis_summary.get('best_blosum') != 'N/A':
            html_body += f"""
            <div class="metric-leaders">
                <h4>🎯 Biological Metric Leaders:</h4>
                <strong>Best Substitution Quality:</strong> {analysis_summary['best_blosum']} (BLOSUM: {analysis_summary['blosum_score']})<br>
                <strong>Best Conservation:</strong> {analysis_summary['best_conservation']} (Entropy: {analysis_summary['entropy_score']})<br>
                <strong>Best Gap Management:</strong> {analysis_summary['best_gaps']} (Gap Fraction: {analysis_summary['gap_score']})
            </div>
            """
        
        html_body += f"""
            <div class="biological-focus">
                <strong>🔬 Analysis Focus:</strong> This analysis prioritizes biological alignment quality over computational efficiency. 
                Rankings are based on substitution quality, sequence conservation, gap management, and sequence similarity.
            </div>
            
            <h3>📊 Results Summary</h3>
            <p>Comprehensive biological quality analysis of <strong>{analysis_summary.get('total_tools', len(results))}</strong> MSA tools.</p>
            
            <table class="results-table">
                <tr>
                    <th>Tool</th>
                    <th>BLOSUM Score<br><small>(higher better)</small></th>
                    <th>Entropy<br><small>(lower better)</small></th>
                    <th>Gap Fraction<br><small>(lower better)</small></th>
                    <th>Percent Identity (%)<br><small>(higher better)</small></th>
                    <th>CPU Time (sec)<br><small>*reference only</small></th>
                    <th>Memory Usage (MB)<br><small>*reference only</small></th>
                </tr>
        """
        
        for result in results:
            if hasattr(result, 'dict'):
                r = result.dict()
            else:
                r = result
                
            # Highlight the best biological tool
            tool_class = 'best-bio-tool' if r['tool'] == analysis_summary.get('best_biological_tool') else ''
            
            html_body += f"""
                <tr>
                    <td class="{tool_class}"><strong>{r['tool']}</strong></td>
                    <td>{r['blosum_score']:,}</td>
                    <td>{r['entropy']:.4f}</td>
                    <td>{r['gap_fraction']:.4f}</td>
                    <td>{r.get('percent_identity', 0):.2f}%</td>
                    <td style="color: #666; font-style: italic;">{r['cpu_time_sec']:.2f}</td>
                    <td style="color: #666; font-style: italic;">{r['memory_usage_mb']:.2f}</td>
                </tr>
            """
        
        html_body += "</table>"
        
        # Add biological recommendations if available
        if analysis_summary.get('recommendations'):
            html_body += """
            <div class="recommendations">
                <h4>🎯 Biological Quality Recommendations:</h4>
                <ul>
            """
            for category, recommendation in analysis_summary['recommendations'].items():
                html_body += f"<li><strong>{category}:</strong> {recommendation}</li>"
            html_body += "</ul></div>"
        
        html_body += """
            <div class="metric-explanation">
                <h4>📈 Biological Quality Metrics:</h4>
                <ul>
                    <li><strong>BLOSUM Score:</strong> Measures amino acid substitution quality using BLOSUM62 matrix - higher scores indicate better biological accuracy in alignments</li>
                    <li><strong>Entropy:</strong> Quantifies conservation at each alignment position - lower entropy indicates better sequence conservation and alignment quality</li>
                    <li><strong>Gap Fraction:</strong> Proportion of gaps in the alignment - lower fractions generally indicate more compact, biologically meaningful alignments</li>
                    <li><strong>Percent Identity:</strong> Average pairwise sequence identity - higher percentages indicate better preservation of sequence similarity</li>
                    <li><strong>CPU Time & Memory:</strong> <em>Computational efficiency metrics provided for reference only - not used in biological quality rankings</em></li>
                </ul>
            </div>
            
            <div class="attachments">
                <h4>📎 Attachments Included:</h4>
                <ul>
                    <li><strong>msa_results_*.csv</strong> - Raw evaluation results in CSV format</li>
        """
        
        if len(results) > 1:
            html_body += """
                    <li><strong>msa_biological_analysis_*.md</strong> - Comprehensive biological quality analysis report with detailed tool comparisons, rankings, and biological insights</li>
                    <li><strong>msa_biological_chart_*.png</strong> - Multi-panel biological quality visualization including rankings, radar charts, and biological metric correlations</li>
            """
        
        html_body += """
                    <li><strong>*_alignment_*.fasta</strong> - Individual alignment files for each tool in FASTA format</li>
                </ul>
            </div>
            
            <div class="footer">
                <p><strong>🧬 Biological Focus:</strong> This analysis prioritizes biological alignment quality over computational speed. Use the biological analysis report for detailed insights into alignment accuracy and biological significance.</p>
                <p><strong>📊 Visualization:</strong> The biological chart provides comprehensive visual analysis of substitution quality, conservation patterns, gap management, and sequence similarity.</p>
                <p>This email was automatically generated by the Enhanced MSA Biological Quality Evaluation Service.</p>
                <p>For questions or support, please contact your system administrator.</p>
            </div>
        </body>
        </html>
        """
        
        return html_body


# Integration function for the FastAPI application
def send_enhanced_results_email(recipient_email: str, results: List[Any], 
                              tmpdir: str, session_id: str, tools_used: List[str],
                              include_analysis: bool = True) -> bool:
    """
    Enhanced integration function to send evaluation results with comprehensive biological analysis via email
    
    Args:
        recipient_email: User's email address
        results: List of evaluation results
        tmpdir: Temporary directory containing alignment files
        session_id: Session identifier
        tools_used: List of tools that were used
        include_analysis: Whether to include comprehensive analysis (default: True)
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Initialize enhanced email service with biological analyzer
        email_service = EnhancedEmailService()
        
        # Collect alignment files
        alignment_files = {}
        for tool in tools_used:
            file_path = os.path.join(tmpdir, f"aligned_{tool}.fasta")
            if os.path.exists(file_path):
                alignment_files[tool] = file_path
        
        # Convert results to dict format for email service
        results_dict = []
        for result in results:
            if hasattr(result, 'dict'):
                results_dict.append(result.dict())
            elif hasattr(result, '__dict__'):
                results_dict.append(result.__dict__)
            else:
                results_dict.append(result)
        
        # Send enhanced email with biological analysis
        success = email_service.send_evaluation_results(
            recipient_email=recipient_email,
            results=results_dict,
            alignment_files=alignment_files,
            session_id=session_id,
            sequence_info=f"Biological quality analysis completed for {len(tools_used)} tools: {', '.join(tools_used)}",
            include_analysis=include_analysis
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Error in send_enhanced_results_email: {str(e)}")
        return False


# Backward compatibility - keep the original function name
def send_results_email(recipient_email: str, results: List[Any], 
                      tmpdir: str, session_id: str, tools_used: List[str]) -> bool:
    """
    Original function for backward compatibility
    """
    return send_enhanced_results_email(recipient_email, results, tmpdir, session_id, tools_used, include_analysis=True)