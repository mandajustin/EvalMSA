import smtplib
import os
import csv
import tempfile
import base64
import io
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

# Import the MSA analyzer from the first file
from msa_analyzer import MSAPerformanceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEmailService:
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587, 
                 sender_email: str = None, sender_password: str = None):
        """
        Initialize enhanced email service with SMTP configuration and MSA analysis
        
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
        
        # Initialize MSA analyzer with balanced weights
        self.msa_analyzer = MSAPerformanceAnalyzer({
            'alignment_quality': 0.5,
            'gap_management': 0.2,
            'computational_efficiency': 0.3
        })

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

    def create_analysis_report(self, results: List[Dict[str, Any]], session_id: str) -> str:
        """
        Create comprehensive MSA analysis report
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            
        Returns:
            Path to the created report file
        """
        try:
            # Convert results to the format expected by MSA analyzer
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
            
            # Generate comprehensive analysis report
            report_content = self.msa_analyzer.analyze_and_report(msa_data, has_percent_identity=True)
            
            # Save report to temporary file
            report_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_analysis_report_{session_id}.md', 
                                                      delete=False, encoding='utf-8')
            report_file.write(report_content)
            report_file.close()
            
            logger.info(f"Analysis report created: {report_file.name}")
            return report_file.name
            
        except Exception as e:
            logger.error(f"Failed to create analysis report: {str(e)}")
            return None

    def create_visualization_chart(self, results: List[Dict[str, Any]], session_id: str) -> str:
        """
        Create comprehensive visualization chart
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            
        Returns:
            Path to the created chart file
        """
        try:
            # Convert results to the format expected by MSA analyzer
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
            
            # Load data and create visualization
            self.msa_analyzer.load_data(msa_data, has_percent_identity=True)
            fig = self.msa_analyzer.create_comprehensive_visualization()
            
            # Save chart to temporary file
            chart_file = tempfile.NamedTemporaryFile(suffix=f'_analysis_chart_{session_id}.png', 
                                                     delete=False)
            fig.savefig(chart_file.name, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Free up memory
            
            logger.info(f"Visualization chart created: {chart_file.name}")
            return chart_file.name
            
        except Exception as e:
            logger.error(f"Failed to create visualization chart: {str(e)}")
            return None

    def get_analysis_summary(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Get a quick analysis summary for email body
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with analysis summary
        """
        try:
            # Convert results to MSA analyzer format
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
            
            # Load data and get recommendations
            self.msa_analyzer.load_data(msa_data, has_percent_identity=True)
            rankings = self.msa_analyzer.rank_tools()
            use_cases = self.msa_analyzer.generate_use_case_recommendations()
            
            # Get top performer
            best_tool = rankings.index[0]
            best_score = rankings.loc[best_tool, 'Overall_Performance']
            
            return {
                'best_tool': best_tool,
                'best_score': f"{best_score:.3f}",
                'total_tools': len(rankings),
                'use_cases': use_cases
            }
            
        except Exception as e:
            logger.error(f"Failed to get analysis summary: {str(e)}")
            return {
                'best_tool': 'N/A',
                'best_score': 'N/A',
                'total_tools': len(results),
                'use_cases': {}
            }

    def send_evaluation_results(self, recipient_email: str, results: List[Dict[str, Any]], 
                              alignment_files: Dict[str, str], session_id: str, 
                              sequence_info: str = "", include_analysis: bool = True) -> bool:
        """
        Send evaluation results with comprehensive analysis via email
        
        Args:
            recipient_email: Recipient's email address
            results: List of evaluation results
            alignment_files: Dictionary mapping tool names to alignment file paths
            session_id: Session identifier
            sequence_info: Optional information about the sequences
            include_analysis: Whether to include comprehensive analysis (default: True)
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Create message container
            msg = MIMEMultipart('mixed')
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"MSA Evaluation Results with Analysis - Session {session_id[:8]}"
            
            # Get analysis summary for email body
            analysis_summary = self.get_analysis_summary(results) if include_analysis else {}
            
            # Create email body
            body = self._create_enhanced_email_body(results, session_id, sequence_info, analysis_summary)
            msg.attach(MIMEText(body, 'html'))
            
            # Create and attach CSV file with results
            csv_file_path = self.create_results_csv(results, session_id)
            self._attach_file(msg, csv_file_path, f"msa_results_{session_id[:8]}.csv")
            
            # Create and attach comprehensive analysis report
            if include_analysis and len(results) > 1:
                report_file_path = self.create_analysis_report(results, session_id)
                if report_file_path:
                    self._attach_file(msg, report_file_path, f"msa_analysis_report_{session_id[:8]}.md")
                
                # Create and attach visualization chart
                chart_file_path = self.create_visualization_chart(results, session_id)
                if chart_file_path:
                    self._attach_file(msg, chart_file_path, f"msa_analysis_chart_{session_id[:8]}.png")
            
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
            
            logger.info(f"Enhanced email with analysis sent successfully to {recipient_email}")
            
            # Clean up temporary files
            try:
                os.unlink(csv_file_path)
                if include_analysis:
                    if 'report_file_path' in locals() and report_file_path:
                        os.unlink(report_file_path)
                    if 'chart_file_path' in locals() and chart_file_path:
                        os.unlink(chart_file_path)
            except OSError as e:
                logger.warning(f"Could not delete temporary files: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send enhanced email: {str(e)}")
            return False

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
        Create enhanced HTML email body with analysis summary
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            sequence_info: Information about the sequences
            analysis_summary: Analysis summary from MSA analyzer
            
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
                .analysis-highlight {{ 
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white; padding: 15px; border-radius: 8px; margin: 15px 0;
                    text-align: center; font-size: 18px; font-weight: bold;
                }}
                .results-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .results-table th, .results-table td {{ 
                    border: 1px solid #ddd; padding: 10px; text-align: center; 
                }}
                .results-table th {{ 
                    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                    color: white; font-weight: bold;
                }}
                .results-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .results-table tr:hover {{ background-color: #e8f5e8; }}
                .recommendations {{ 
                    background-color: #f0f8ff; padding: 15px; border-radius: 8px; 
                    border-left: 5px solid #4CAF50; margin: 20px 0;
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
                .best-tool {{ color: #28a745; font-weight: bold; font-size: 1.1em; }}
                ul {{ padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🧬 Multiple Sequence Alignment Evaluation Results</h2>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                {f'<p><strong>Sequence Info:</strong> {sequence_info}</p>' if sequence_info else ''}
            </div>
        """
        
        # Add analysis highlight if available
        if analysis_summary.get('best_tool') != 'N/A':
            html_body += f"""
            <div class="analysis-highlight">
                🏆 RECOMMENDED TOOL: {analysis_summary['best_tool'].upper()} 
                (Score: {analysis_summary['best_score']})
            </div>
            """
        
        html_body += f"""
            <h3>📊 Results Summary</h3>
            <p>Comprehensive analysis of <strong>{analysis_summary.get('total_tools', len(results))}</strong> MSA tools across multiple performance criteria.</p>
            
            <table class="results-table">
                <tr>
                    <th>Tool</th>
                    <th>BLOSUM Score</th>
                    <th>Entropy</th>
                    <th>Gap Fraction</th>
                    <th>Percent Identity (%)</th>
                    <th>CPU Time (sec)</th>
                    <th>Memory Usage (MB)</th>
                </tr>
        """
        
        for result in results:
            if hasattr(result, 'dict'):
                r = result.dict()
            else:
                r = result
                
            # Highlight the best tool
            tool_class = 'best-tool' if r['tool'] == analysis_summary.get('best_tool') else ''
            
            html_body += f"""
                <tr>
                    <td class="{tool_class}"><strong>{r['tool']}</strong></td>
                    <td>{r['blosum_score']:,}</td>
                    <td>{r['entropy']:.4f}</td>
                    <td>{r['gap_fraction']:.4f}</td>
                    <td>{r.get('percent_identity', 0):.2f}%</td>
                    <td>{r['cpu_time_sec']:.2f}</td>
                    <td>{r['memory_usage_mb']:.2f}</td>
                </tr>
            """
        
        html_body += "</table>"
        
        # Add use case recommendations if available
        if analysis_summary.get('use_cases'):
            html_body += """
            <div class="recommendations">
                <h4>🎯 Use Case Specific Recommendations:</h4>
                <ul>
            """
            for use_case, recommendation in analysis_summary['use_cases'].items():
                html_body += f"<li><strong>{use_case}:</strong> {recommendation}</li>"
            html_body += "</ul></div>"
        
        html_body += """
            <div class="metric-explanation">
                <h4>📈 Metric Explanations:</h4>
                <ul>
                    <li><strong>BLOSUM Score:</strong> Sum of amino acid substitution scores using BLOSUM62 matrix (higher is better for accuracy)</li>
                    <li><strong>Entropy:</strong> Average column-wise entropy measuring alignment conservation (lower indicates better conservation)</li>
                    <li><strong>Gap Fraction:</strong> Proportion of gaps in the alignment (lower is generally better for compactness)</li>
                    <li><strong>Percent Identity:</strong> Average pairwise sequence identity across all sequence pairs (higher indicates more similarity)</li>
                    <li><strong>CPU Time:</strong> Computational time required for alignment (lower is better for efficiency)</li>
                    <li><strong>Memory Usage:</strong> Peak memory consumption during alignment (lower is better for resource efficiency)</li>
                </ul>
            </div>
            
            <div class="attachments">
                <h4>📎 Attachments Included:</h4>
                <ul>
                    <li><strong>msa_results_*.csv</strong> - Raw evaluation results in CSV format for further analysis</li>
        """
        
        if len(results) > 1:
            html_body += """
                    <li><strong>msa_analysis_report_*.md</strong> - Comprehensive analysis report with detailed tool comparisons and recommendations</li>
                    <li><strong>msa_analysis_chart_*.png</strong> - Multi-panel visualization including performance rankings, radar charts, and statistical plots</li>
            """
        
        html_body += """
                    <li><strong>*_alignment_*.fasta</strong> - Individual alignment files for each tool in FASTA format</li>
                </ul>
            </div>
            
            <div class="footer">
                <p><strong>💡 Pro Tip:</strong> Open the analysis report (Markdown file) in any text editor or Markdown viewer for detailed insights, and view the chart (PNG file) for visual comparisons.</p>
                <p>This email was automatically generated by the Enhanced MSA Evaluation Service with integrated performance analysis.</p>
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
    Enhanced integration function to send evaluation results with comprehensive analysis via email
    
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
        # Initialize enhanced email service
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
        
        # Send enhanced email with analysis
        success = email_service.send_evaluation_results(
            recipient_email=recipient_email,
            results=results_dict,
            alignment_files=alignment_files,
            session_id=session_id,
            sequence_info=f"Analysis completed for {len(tools_used)} tools: {', '.join(tools_used)}",
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