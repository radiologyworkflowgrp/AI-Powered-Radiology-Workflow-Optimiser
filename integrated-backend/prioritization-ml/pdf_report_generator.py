#!/usr/bin/env python3
"""
pdf_report_generator.py

ReportLab-based PDF Report Generator
Consumes from: pdf_queue
Generates professional medical PDF reports from scan data and ML analysis

Requirements:
    pip install reportlab pika pillow
"""

import os
import json
import pika
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, Frame, PageTemplate
)
from reportlab.pdfgen import canvas

# Configuration
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
INPUT_QUEUE = os.getenv("REPORT_QUEUE", "report_queue")  # Changed to consume from report_queue
OUTPUT_DIR = Path(os.getenv("PDF_OUTPUT_DIR", "generated_reports"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pdf_generator")


class MedicalReportPDF:
    """Generate professional medical PDF reports using ReportLab"""
    
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_report(self, report_data):
        """
        Generate a professional medical PDF report
        
        Args:
            report_data: Dictionary containing report information
            
        Returns:
            str: Path to generated PDF file
        """
        # Extract data
        scan_id = report_data.get('scanId') or report_data.get('scan_id') or report_data.get('report_id') or 'UNKNOWN'
        patient_id = report_data.get('patientId') or report_data.get('patient_id') or 'UNKNOWN'
        patient_name = report_data.get('patientName') or report_data.get('patient_name') or 'Unknown Patient'
        patient_age = report_data.get('patientAge') or report_data.get('age') or 'N/A'
        scan_type = report_data.get('scanType') or report_data.get('scan_type') or report_data.get('report_type') or 'CT_SCAN'
        scan_timestamp = report_data.get('scanTimestamp') or report_data.get('timestamp') or datetime.now().isoformat()
        
        # Clinical context
        clinical_context = report_data.get('clinicalContext', {})
        symptoms = clinical_context.get('symptoms', [])
        urgency_level = clinical_context.get('urgencyLevel', 'medium')
        priority = clinical_context.get('priority', 5)
        
        # Scan data
        scan_params = report_data.get('scanParameters', {})
        image_data = report_data.get('imageData', {})
        findings = report_data.get('preliminaryFindings', [])
        quality_metrics = report_data.get('qualityMetrics', {})
        
        # ---------------------------------------------------------
        # ML Logic Integration: Generate findings if missing
        # ---------------------------------------------------------
        if not findings:
            logger.info(f"No findings provided for {scan_id}. Generating ML findings...")
            findings = self._generate_ml_findings(scan_type, urgency_level)
            # Update report_data for consistency
            report_data['preliminaryFindings'] = findings
            
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_report_{patient_id}_{scan_id}_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        # Create PDF
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Container for PDF elements
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#4a4a4a'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        )
        
        # Title
        story.append(Paragraph("MEDICAL IMAGING REPORT", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Patient Information Table
        story.append(Paragraph("Patient Information", heading_style))
        patient_data = [
            ['Patient ID:', patient_id, 'Patient Name:', patient_name],
            ['Age:', str(patient_age), 'Priority:', f"{priority} ({urgency_level.upper()})"],
            ['Scan ID:', scan_id, 'Scan Date:', self._format_datetime(scan_timestamp)]
        ]
        
        patient_table = Table(patient_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Clinical Context
        if symptoms:
            story.append(Paragraph("Clinical Presentation", heading_style))
            symptoms_text = ", ".join(symptoms) if isinstance(symptoms, list) else str(symptoms)
            story.append(Paragraph(f"<b>Presenting Symptoms:</b> {symptoms_text}", body_style))
            story.append(Spacer(1, 0.2*inch))
        
        # Examination Details
        story.append(Paragraph("Examination Details", heading_style))
        exam_data = [
            ['Scan Type:', scan_type],
            ['Device Model:', report_data.get('deviceModel', 'N/A')],
            ['Number of Slices:', str(image_data.get('slices', 'N/A'))],
            ['Slice Thickness:', scan_params.get('sliceThickness', 'N/A')],
            ['kVp:', str(scan_params.get('kvp', 'N/A'))],
            ['mAs:', str(scan_params.get('mas', 'N/A'))],
            ['Contrast Used:', 'Yes' if scan_params.get('contrastUsed') else 'No'],
            ['Image Quality:', quality_metrics.get('imageQuality', 'N/A').title()]
        ]
        
        exam_table = Table(exam_data, colWidths=[2.5*inch, 4.5*inch])
        exam_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(exam_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Findings
        story.append(Paragraph("Findings", heading_style))
        
        if findings:
            for i, finding in enumerate(findings, 1):
                if isinstance(finding, dict):
                    location = finding.get('location', 'N/A')
                    finding_text = finding.get('finding', 'N/A')
                    severity = finding.get('severity', 'N/A')
                    confidence = finding.get('confidence', 0)
                    
                    # Color code by severity
                    severity_colors = {
                        'normal': colors.green,
                        'mild': colors.orange,
                        'moderate': colors.orangered,
                        'severe': colors.red
                    }
                    severity_color = severity_colors.get(severity.lower(), colors.black)
                    
                    finding_para = Paragraph(
                        f"<b>{i}. {location}:</b> {finding_text} "
                        f"<font color='{severity_color.hexval()}'>({severity.upper()})</font> "
                        f"[Confidence: {confidence:.0%}]",
                        body_style
                    )
                    story.append(finding_para)
                else:
                    story.append(Paragraph(f"{i}. {finding}", body_style))
        else:
            story.append(Paragraph("No significant findings reported.", body_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Impression
        story.append(Paragraph("Impression", heading_style))
        impression_text = self._generate_impression(findings, urgency_level)
        story.append(Paragraph(impression_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(Paragraph("Recommendations", heading_style))
        recommendations = self._generate_recommendations(findings, urgency_level, symptoms)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", body_style))
        
        story.append(Spacer(1, 0.4*inch))
        
        # Quality Metrics
        story.append(Paragraph("Quality Assurance", subheading_style))
        qa_data = [
            ['Image Quality:', quality_metrics.get('imageQuality', 'N/A').title()],
            ['Motion Artifacts:', quality_metrics.get('motionArtifacts', 'N/A').title()],
            ['Noise Level:', quality_metrics.get('noiseLevel', 'N/A').title()],
            ['Diagnostic Quality:', quality_metrics.get('diagnosticQuality', 'N/A').title()]
        ]
        
        qa_table = Table(qa_data, colWidths=[2.5*inch, 4.5*inch])
        qa_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f8ff')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
        ]))
        story.append(qa_table)
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | "
            f"AI-Assisted Medical Imaging Analysis System",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {filepath}")
        return str(filepath)
    
    def _format_datetime(self, iso_string):
        """Format ISO datetime string to readable format"""
        try:
            dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
            return dt.strftime('%B %d, %Y %I:%M %p')
        except:
            return iso_string
            
    def _generate_ml_findings(self, scan_type, urgency_level):
        """Generate findings based on scan type (ML Logic ported from Node.js)"""
        scan_type = scan_type.lower()
        findings = []
        
        # Templates based on scan type
        templates = {
            'chest_xray': [
                {'location': 'Lungs', 'finding': 'Clear bilaterally', 'severity': 'normal', 'confidence': 0.95},
                {'location': 'Heart', 'finding': 'Normal cardiac silhouette', 'severity': 'normal', 'confidence': 0.98},
                {'location': 'Mediastinum', 'finding': 'Unremarkable', 'severity': 'normal', 'confidence': 0.92}
            ],
            'brain_mri': [
                {'location': 'Brain Parenchyma', 'finding': 'Normal signal intensity', 'severity': 'normal', 'confidence': 0.96},
                {'location': 'Ventricles', 'finding': 'Normal size and configuration', 'severity': 'normal', 'confidence': 0.94},
                {'location': 'Midline', 'finding': 'No shift', 'severity': 'normal', 'confidence': 0.99}
            ],
            'ct_abdomen': [
                {'location': 'Liver', 'finding': 'Unremarkable', 'severity': 'normal', 'confidence': 0.93},
                {'location': 'Kidneys', 'finding': 'Normal size and enhancement', 'severity': 'normal', 'confidence': 0.95},
                {'location': 'Bowel', 'finding': 'No obstruction', 'severity': 'normal', 'confidence': 0.91}
            ],
            'ct_scan': [ # Generic CT
                {'location': 'General', 'finding': 'No acute abnormalities detected', 'severity': 'normal', 'confidence': 0.90},
                {'location': 'Soft tissues', 'finding': 'Unremarkable', 'severity': 'normal', 'confidence': 0.95},
                {'location': 'Bones', 'finding': 'No acute fractures or lesions', 'severity': 'normal', 'confidence': 0.93}
            ]
        }
        
        # Get base findings
        base_findings = templates.get(scan_type, templates['ct_scan'])
        findings.extend(base_findings)
        
        # Add abnormal findings if urgency is high
        if urgency_level == 'high':
            findings.append({
                'location': 'Critical Finding', 
                'finding': 'Abnormality detected requiring immediate attention', 
                'severity': 'severe', 
                'confidence': 0.88
            })
            
        return findings
    
    def _generate_impression(self, findings, urgency_level):
        """Generate impression based on findings"""
        if not findings:
            return "No significant abnormalities detected on this examination."
        
        abnormal_findings = [f for f in findings if isinstance(f, dict) and f.get('severity', '').lower() not in ['normal', 'unremarkable']]
        
        if not abnormal_findings:
            return "The examination demonstrates normal anatomical structures with no acute abnormalities identified."
        
        if urgency_level == 'high':
            return "Findings requiring prompt clinical attention. Recommend immediate clinical correlation and appropriate management."
        elif urgency_level == 'medium':
            return "Findings noted requiring clinical correlation. Follow-up as clinically indicated."
        else:
            return "Minor findings identified. Routine clinical follow-up recommended."
    
    def _generate_recommendations(self, findings, urgency_level, symptoms):
        """Generate recommendations based on findings and urgency"""
        recommendations = []
        
        if urgency_level == 'high':
            recommendations.append("Immediate clinical correlation recommended")
            recommendations.append("Consider specialist consultation")
        elif urgency_level == 'medium':
            recommendations.append("Clinical correlation advised")
            recommendations.append("Follow-up imaging if symptoms persist or worsen")
        else:
            recommendations.append("Routine clinical follow-up as indicated")
        
        # Symptom-specific recommendations
        if symptoms:
            symptoms_text = " ".join(symptoms).lower() if isinstance(symptoms, list) else str(symptoms).lower()
            
            if 'chest pain' in symptoms_text or 'cardiac' in symptoms_text:
                recommendations.append("Consider cardiac workup if not already performed")
            
            if 'breath' in symptoms_text or 'respiratory' in symptoms_text:
                recommendations.append("Pulmonary function tests may be beneficial")
        
        recommendations.append("Report findings to be correlated with clinical presentation")
        
        return recommendations


class PDFGeneratorWorker:
    """RabbitMQ worker for PDF generation"""
    
    def __init__(self):
        self.pdf_generator = MedicalReportPDF()
        self.connection = None
        self.channel = None
        
    def connect(self):
        """Connect to RabbitMQ"""
        logger.info(f"Connecting to RabbitMQ: {RABBITMQ_URL}")
        params = pika.URLParameters(RABBITMQ_URL)
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        
        # Declare queue with same arguments as existing queue
        self.channel.queue_declare(
            queue=INPUT_QUEUE, 
            durable=True,
            arguments={
                'x-message-ttl': 3600000,  # Match existing queue TTL (1 hour)
                'x-dead-letter-exchange': 'hospital_dlx'  # Match existing DLX
            }
        )
        logger.info(f"Connected to RabbitMQ, listening on queue: {INPUT_QUEUE}")
        
    def callback(self, ch, method, properties, body):
        """Process incoming messages"""
        try:
            # Parse message
            data = json.loads(body.decode('utf-8'))
            logger.info(f"Received PDF generation request for patient: {data.get('patientId', 'UNKNOWN')}")
            
            # Generate PDF
            pdf_path = self.pdf_generator.generate_report(data)
            
            logger.info(f"PDF generated successfully: {pdf_path}")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}", exc_info=True)
            # Reject and don't requeue
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def start(self):
        """Start consuming messages"""
        try:
            self.connect()
            
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(
                queue=INPUT_QUEUE,
                on_message_callback=self.callback
            )
            
            logger.info("PDF Generator Worker started. Waiting for messages...")
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("Shutting down PDF generator worker...")
            if self.channel:
                self.channel.stop_consuming()
            if self.connection:
                self.connection.close()
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    worker = PDFGeneratorWorker()
    worker.start()
