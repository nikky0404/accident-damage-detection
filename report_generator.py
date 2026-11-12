"""
Automated Vehicle Damage Assessment Report Generator.
Generates publication-quality PDF reports (via ReportLab) and structured JSON exports.
"""

import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_pdf(self, report_data):
        """
        Creates a professional PDF inspection report.
        report_data format:
        {
            "inspection_id": int/str,
            "date": str,
            "vehicle": {"manufacturer": str, "model": str, "year": int, "registration_number": str},
            "summary": {"image_count": int, "damaged_parts": int, "severity_counts": dict},
            "items": list of dicts,
            "cost": {"parts": float, "labor": float, "total": float, "min_range": float, "max_range": float}
        }
        Returns:
            pdf_path: absolute path to generated PDF
        """
        insp_id = report_data.get("inspection_id", "INSP-001")
        filename = f"damage_assessment_report_{insp_id}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        normal_style = styles["Normal"]

        # Custom professional styling
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            fontName="Helvetica-Bold",
            alignment=0
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#0284c7"),
            fontName="Helvetica-Bold"
        )
        heading2_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            fontName="Helvetica-Bold",
            spaceAfter=6
        )

        story = []

        # Top Header Banner
        header_table_data = [
            [
                Paragraph("<b>360° AUTO VISION</b><br/><font size='9' color='#64748b'>AI-Powered Automotive Inspection & Damage Assessment</font>", normal_style),
                Paragraph(f"<b>INSPECTION ID:</b> #{insp_id}<br/><b>DATE:</b> {report_data.get('date', datetime.utcnow().strftime('%Y-%m-%d'))}", normal_style)
            ]
        ]
        header_table = Table(header_table_data, colWidths=[340, 200])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (1,0), (1,0), 'RIGHT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8)
        ]))
        story.append(header_table)
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=14))

        # Title
        story.append(Paragraph("VEHICLE DAMAGE ASSESSMENT REPORT", title_style))
        story.append(Paragraph("Certified Computer Vision & Deep Learning AI Estimate", subtitle_style))
        story.append(Spacer(1, 14))

        # Vehicle Info & Summary Cards (2 Columns)
        veh = report_data.get("vehicle", {})
        summ = report_data.get("summary", {})

        veh_info_text = (
            f"<b>Manufacturer:</b> {veh.get('manufacturer', 'N/A')}<br/>"
            f"<b>Model:</b> {veh.get('model', 'N/A')}<br/>"
            f"<b>Manufacturing Year:</b> {veh.get('year', 'N/A')}<br/>"
            f"<b>Registration Number:</b> {veh.get('registration_number', 'N/A')}"
        )

        sev = summ.get("severity_counts", {})
        summary_text = (
            f"<b>Images Analyzed:</b> {summ.get('image_count', 0)} (360° Multi-View)<br/>"
            f"<b>Damaged Components:</b> {summ.get('damaged_parts', 0)}<br/>"
            f"<b>Minor Damage:</b> {sev.get('Minor', 0)} | <b>Moderate:</b> {sev.get('Moderate', 0)} | <b>Severe:</b> {sev.get('Severe', 0)}<br/>"
            f"<b>AI Fusion Status:</b> Multi-angle duplicates consolidated"
        )

        overview_data = [
            [
                Paragraph("<b>VEHICLE DETAILS</b>", heading2_style),
                Paragraph("<b>INSPECTION SUMMARY</b>", heading2_style)
            ],
            [
                Paragraph(veh_info_text, normal_style),
                Paragraph(summary_text, normal_style)
            ]
        ]
        overview_table = Table(overview_data, colWidths=[270, 270])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 16))

        # Itemized Damaged Components Table
        story.append(Paragraph("DAMAGED COMPONENTS & REPAIR BREAKDOWN", heading2_style))

        items_header = ["Vehicle Part", "Damage Type", "Severity", "Area %", "Action", "Parts Cost", "Labor", "Total"]
        items_rows = [items_header]

        for it in report_data.get("items", []):
            part = str(it.get("part", "Panel"))
            dtype = str(it.get("damage_type", "Dent"))
            sev_str = str(it.get("severity", "Moderate"))
            area_str = f"{it.get('damaged_area', 0.0)}%"
            act = str(it.get("action", "Repair"))
            p_cost = f"₹{int(it.get('parts_cost', 0)):,}"
            l_cost = f"₹{int(it.get('labor_cost', 0)):,}"
            tot_cost = f"₹{int(it.get('estimated_cost', 0)):,}"

            items_rows.append([part, dtype, sev_str, area_str, act, p_cost, l_cost, tot_cost])

        comp_table = Table(items_rows, colWidths=[100, 75, 60, 50, 75, 60, 55, 65])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('ALIGN', (3,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,1), (-1,-1), 8.5),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 16))

        # Cost Summary Box
        cost = report_data.get("cost", {})
        tot_val = int(cost.get("total", 0))
        min_v = int(cost.get("min_range", tot_val * 0.88))
        max_v = int(cost.get("max_range", tot_val * 1.14))

        cost_summary_data = [
            [
                Paragraph("<b>Total Parts Cost:</b>", normal_style),
                Paragraph(f"<b>₹{int(cost.get('parts', 0)):,}</b>", normal_style),
                Paragraph("<b>ESTIMATED TOTAL REPAIR COST:</b>", heading2_style),
                Paragraph(f"<font size='14' color='#0284c7'><b>₹{tot_val:,}</b></font>", normal_style)
            ],
            [
                Paragraph("<b>Total Labor Cost:</b>", normal_style),
                Paragraph(f"<b>₹{int(cost.get('labor', 0)):,}</b>", normal_style),
                Paragraph("<b>Estimated Range (95% CI):</b>", normal_style),
                Paragraph(f"<b>₹{min_v:,} – ₹{max_v:,}</b>", normal_style)
            ]
        ]
        cost_table = Table(cost_summary_data, colWidths=[120, 100, 180, 140])
        cost_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0f9ff")),
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#0284c7")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8)
        ]))
        story.append(cost_table)
        story.append(Spacer(1, 20))

        # Disclaimer & Academic Notice
        disclaimer_text = (
            "<font size='7.5' color='#64748b'><b>DISCLAIMER:</b> This assessment is generated using Computer Vision (YOLOv8/11, Segmentation) "
            "and Machine Learning (XGBoost / Random Forest) predictive models. The estimated repair costs represent regional benchmark "
            "approximations and may vary based on authorized dealership quotations, OEM parts availability, and internal mechanical damages "
            "hidden beneath body panels.</font><br/><br/>"
            "<font size='7' color='#94a3b8'>Project developed for Bharath Institute of Higher Education and Research – 360° Auto Vision Platform.</font>"
        )
        story.append(Paragraph(disclaimer_text, normal_style))

        doc.build(story)
        return filepath

    def generate_json(self, report_data):
        """Generates JSON string and file for export."""
        insp_id = report_data.get("inspection_id", "INSP-001")
        filename = f"damage_assessment_report_{insp_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        return filepath
