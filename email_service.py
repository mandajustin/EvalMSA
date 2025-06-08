import smtplib
import os
import csv
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any
import logging
from datetime import datetime
from models import EvalResult


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587, 
                 sender_email: str = None, sender_password: str = None):
        """
        Initialize email service with SMTP configuration
        
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
        
        if not self.sender_email or not self.sender_password:
            raise ValueError("Sender email and password must be provided via parameters or environment variables")

    def create_results_csv(self, results: List[Dict[str, Any]], session_id: str) -> str:
        """
        Create a CSV file from evaluation results
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            
        Returns:
            Path to the created CSV file
        """
        # Create temporary CSV file
        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_results_{session_id}.csv', 
                                               delete=False, newline='')
        
        if not results:
            logger.warning("No results to write to CSV")
            return csv_file.name
            
        # Write CSV headers and data
        fieldnames = ['tool', 'blosum_score', 'entropy', 'gap_fraction', 'cpu_time_sec', 'memory_usage_mb']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            # Handle both dict and object types
            if hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                result_dict = result
            writer.writerow(result_dict)
        
        csv_file.close()
        logger.info(f"CSV file created: {csv_file.name}")
        return csv_file.name

    def send_evaluation_results(self, recipient_email: str, results: List[Dict[str, Any]], 
                              alignment_files: Dict[str, str], session_id: str, 
                              sequence_info: str = "") -> bool:
        """
        Send evaluation results and alignment files via email
        
        Args:
            recipient_email: Recipient's email address
            results: List of evaluation results
            alignment_files: Dictionary mapping tool names to alignment file paths
            session_id: Session identifier
            sequence_info: Optional information about the sequences
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Create message container
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"MSA Evaluation Results - Session {session_id[:8]}"
            
            # Create email body
            body = self._create_email_body(results, session_id, sequence_info)
            msg.attach(MIMEText(body, 'html'))
            
            # Create and attach CSV file with results
            csv_file_path = self.create_results_csv(results, session_id)
            self._attach_file(msg, csv_file_path, f"msa_results_{session_id[:8]}.csv")
            
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
            
            logger.info(f"Email sent successfully to {recipient_email}")
            
            # Clean up temporary CSV file
            try:
                os.unlink(csv_file_path)
            except OSError:
                logger.warning(f"Could not delete temporary CSV file: {csv_file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
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

    def _create_email_body(self, results: List[Dict[str, Any]], session_id: str, 
                          sequence_info: str) -> str:
        """
        Create HTML email body with results summary
        
        Args:
            results: List of evaluation results
            session_id: Session identifier
            sequence_info: Information about the sequences
            
        Returns:
            HTML formatted email body
        """
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; }}
                .results-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .results-table th, .results-table td {{ 
                    border: 1px solid #ddd; padding: 8px; text-align: center; 
                }}
                .results-table th {{ background-color: #4CAF50; color: white; }}
                .results-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Multiple Sequence Alignment Evaluation Results</h2>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                {f'<p><strong>Sequence Info:</strong> {sequence_info}</p>' if sequence_info else ''}
            </div>
            
            <h3>Results Summary</h3>
            <table class="results-table">
                <tr>
                    <th>Tool</th>
                    <th>BLOSUM Score</th>
                    <th>Entropy</th>
                    <th>Gap Fraction</th>
                    <th>CPU Time (sec)</th>
                    <th>Memory Usage (MB)</th>
                </tr>
        """
        
        for result in results:
            # Handle both dict and object types
            if hasattr(result, 'dict'):
                r = result.dict()
            else:
                r = result
                
            html_body += f"""
                <tr>
                    <td><strong>{r['tool']}</strong></td>
                    <td>{r['blosum_score']}</td>
                    <td>{r['entropy']:.4f}</td>
                    <td>{r['gap_fraction']:.4f}</td>
                    <td>{r['cpu_time_sec']:.2f}</td>
                    <td>{r['memory_usage_mb']:.2f}</td>
                </tr>
            """
        
        html_body += """
            </table>
            
            <h3>Attachments</h3>
            <ul>
                <li><strong>msa_results_*.csv</strong> - Detailed evaluation results in CSV format</li>
                <li><strong>*_alignment_*.fasta</strong> - Alignment files for each tool in FASTA format</li>
            </ul>
            
            <div class="footer">
                <p>This email was automatically generated by the MSA Evaluation Service.</p>
                <p>For questions or support, please contact your system administrator.</p>
            </div>
        </body>
        </html>
        """
        
        return html_body


# Integration function for the FastAPI application
def send_results_email(recipient_email: str, results: List[EvalResult], 
                      tmpdir: str, session_id: str, tools_used: List[str]) -> bool:
    """
    Integration function to send evaluation results via email
    
    Args:
        recipient_email: User's email address
        results: List of evaluation results
        tmpdir: Temporary directory containing alignment files
        session_id: Session identifier
        tools_used: List of tools that were used
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Initialize email service
        email_service = EmailService()
        
        # Collect alignment files
        alignment_files = {}
        for tool in tools_used:
            file_path = os.path.join(tmpdir, f"aligned_{tool}.fasta")
            if os.path.exists(file_path):
                alignment_files[tool] = file_path
        
        # Convert results to dict format for email service
        results_dict = [result.dict() if hasattr(result, 'dict') else result for result in results]
        
        # Send email
        success = email_service.send_evaluation_results(
            recipient_email=recipient_email,
            results=results_dict,
            alignment_files=alignment_files,
            session_id=session_id,
            sequence_info=f"Analysis completed for {len(tools_used)} tools"
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Error in send_results_email: {str(e)}")
        return False
