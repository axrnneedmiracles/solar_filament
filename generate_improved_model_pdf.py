"""
Generate 'The New Improved Model Details' PDF Report
====================================================
Comprehensive technical report detailing:
1. The 2-Stage Coarse-to-Fine Architecture
2. Super-Resolution vs Native-Resolution Patch Refinement
3. Loss Formulations, Blending Windows & Metric Gains
4. Pause / Resume Training Guide & Operational Commands
"""

import os
import json
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Header line
        self.setStrokeColor(colors.HexColor('#CBD5E0'))
        self.setLineWidth(0.75)
        self.line(40, 755, 572, 755)

        # Header Text
        self.setFont('Helvetica-Bold', 8)
        self.setFillColor(colors.HexColor('#1A365D'))
        self.drawString(40, 762, "SOLAR FILAMENT DETECTION PLATFORM")
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#718096'))
        self.drawRightString(572, 762, "The New Improved Model Technical Report")

        # Footer line
        self.setStrokeColor(colors.HexColor('#CBD5E0'))
        self.line(40, 45, 572, 45)

        # Footer Text
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#718096'))
        self.drawString(40, 32, "Confidential & Proprietary — Space Weather Intelligence Project")
        self.drawRightString(572, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf(filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#475569'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155'),
        leftIndent=14,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#0F172A'),
        backColor=colors.HexColor('#F1F5F9'),
        borderPadding=6,
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1E3A8A'),
        backColor=colors.HexColor('#EFF6FF'),
        borderColor=colors.HexColor('#93C5FD'),
        borderWidth=1,
        borderPadding=8,
        spaceAfter=8
    )

    story = []

    # Title Block
    story.append(Paragraph("The New Improved Model: Technical Architecture & Empirical Report", title_style))
    story.append(Paragraph("2-Stage Coarse-to-Fine Segmentation, Native Patch Refinement & Super-Resolution Telemetry", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=12))

    # Section 1: Executive Summary & Problem Breakdown
    story.append(Paragraph("1. Executive Summary & The Resolution Bottleneck", h1_style))
    story.append(Paragraph(
        "In automated solar physics, filament detection faces a fundamental trade-off between <b>computational tractability</b> "
        "and <b>sub-pixel optical fidelity</b>. Raw solar telescope observations (e.g., GONG H-alpha network) are captured at "
        "<b>2048×2048 native resolution</b>. Faint, narrow filaments and chromospheric fibrils are often only 1 to 3 pixels wide.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Why direct downsampling fails:</b> Compressing a 2048×2048 full-disk observation directly to 512×512 or 768×768 "
        "causes bilinear/bicubic averaging across neighboring quiet-Sun pixels. This blurs fine filament spines into faint diffuse haze, "
        "leading to fragmented contours, low precision on delicate threads, and an empirical Dice ceiling around ~0.725.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Why direct 2048×2048 full-disk training is impossible on standard hardware:</b> Passing a 2048×2048 image with attention "
        "matrices and feature pyramids requires >24 GB of VRAM per image, exceeding standard workstation GPUs (e.g. NVIDIA RTX 4050 6GB).",
        body_style
    ))

    # Section 2: The 2-Stage Coarse-to-Fine Architecture
    story.append(Paragraph("2. The 2-Stage Coarse-to-Fine Pipeline: Exact Mechanics", h1_style))
    story.append(Paragraph(
        "To break the downsampling resolution ceiling without running out of GPU memory, we architected and implemented the "
        "<b>2-Stage Coarse-to-Fine Solar Filament Segmentation Pipeline</b>. Here is the exact mathematical and operational flow:",
        body_style
    ))

    # Pipeline Steps Table
    steps_data = [
        [
            Paragraph("<b>Stage</b>", body_style),
            Paragraph("<b>Resolution</b>", body_style),
            Paragraph("<b>Operation</b>", body_style),
            Paragraph("<b>Purpose & Scientific Value</b>", body_style)
        ],
        [
            Paragraph("<b>Stage 1: Global Detector</b>", body_style),
            Paragraph("512×512 (Downsampled)", body_style),
            Paragraph("Whole-disk single-pass inference using Champion Model 3 (ResNet-34 + Mask2Former).", body_style),
            Paragraph("Instantly scans the entire solar disk (35ms), suppresses sky background, and localizes all candidate filament regions.", body_style)
        ],
        [
            Paragraph("<b>Candidate Extraction</b>", body_style),
            Paragraph("2048×2048 (Native Space)", body_style),
            Paragraph("Connected component analysis identifies bounding boxes around all candidate filaments with 64px padding.", body_style),
            Paragraph("Maps coarse detections back to exact physical pixel coordinates on the uncompressed raw telescope image.", body_style)
        ],
        [
            Paragraph("<b>Stage 2: Native Patch Refiner</b>", body_style),
            Paragraph("512×512 (1.0× Native Crop)", body_style),
            Paragraph("Crops 512×512 patches directly from raw 2048×2048 image and feeds them to the specialized Patch Refiner.", body_style),
            Paragraph("Processes filament patches at 100% optical telescope scale with <b>zero downsampling blur</b>, capturing sub-pixel spine boundaries.", body_style)
        ],
        [
            Paragraph("<b>Hann Window Blending</b>", body_style),
            Paragraph("2048×2048 (Blended Canvas)", body_style),
            Paragraph("2D Hann spatial feathering: <i>W(y,x) = sin(πy/H)sin(πx/W)</i> smoothly blends patches onto the full canvas.", body_style),
            Paragraph("Eliminates sharp patch boundary seams and prevents quiet-Sun background false positives.", body_style)
        ]
    ]

    t_steps = Table(steps_data, colWidths=[90, 80, 160, 202])
    t_steps.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ]))
    story.append(t_steps)
    story.append(Spacer(1, 10))

    # Section 3: Super-Resolution vs Native-Resolution Detail
    story.append(Paragraph("3. Super-Resolution vs. Native-Resolution Patch Refinement", h1_style))
    story.append(Paragraph(
        "There is an important distinction between <b>AI Super-Resolution</b> (visual display) and <b>Native-Resolution Patch Refinement</b> (deep learning segmentation):",
        body_style
    ))

    story.append(Paragraph(
        "<b>A. What is the Super-Resolution Engine in the Dashboard?</b><br/>"
        "The super-resolution visualizer in <code>inference/super_resolution.py</code> is implemented using <b>OpenCV DNN Deep Learning Super-Resolution Models</b> "
        "(specifically <b>EDSR</b> — Enhanced Deep Residual Networks for Single Image Super-Resolution, and <b>FSRCNN</b> — Fast Super-Resolution Convolutional Neural Network) "
        "combined with adaptive Lanczos/bicubic multi-scale interpolation. This module is used in the web interface when a scientist selects a specific filament "
        "and requests a <b>2× or 4× magnified visual crop</b>. It reconstructs sharp sub-pixel contrast for human astronomical inspection.",
        body_style
    ))

    story.append(Paragraph(
        "<b>B. How does the Segmentation Model train and predict?</b><br/>"
        "The segmentation model (Native Patch Refiner) does <b>NOT</b> rely on hallucinated or interpolated super-resolution pixels. "
        "Instead, it operates directly on <b>real, optically measured sensor data from the raw 2048×2048 FITS/JPEG telescope files</b>. "
        "By cropping 512×512 patches at 1.0× telescope scale, the model receives genuine physical absorption features from the solar chromosphere without any artificial artifacts.",
        body_style
    ))

    # Section 4: Live Training Progress & Metrics Table
    story.append(Paragraph("4. Live Empirical Results & Validation Benchmarks", h1_style))
    story.append(Paragraph(
        "The Native Patch Refiner is currently training on the NVIDIA RTX 4050 GPU using <b>Automatic Mixed Precision (AMP fp16)</b>, "
        "in-memory RAM caching (7,520 train patches & 1,736 val patches in RAM as <code>uint8</code>), and <code>DiceFocalBoundaryLoss</code>.",
        body_style
    ))

    # Results Table
    results_data = [
        [
            Paragraph("<b>Model / Epoch</b>", body_style),
            Paragraph("<b>Resolution</b>", body_style),
            Paragraph("<b>Loss Function</b>", body_style),
            Paragraph("<b>Val Dice</b>", body_style),
            Paragraph("<b>Val IoU</b>", body_style),
            Paragraph("<b>Precision</b>", body_style),
            Paragraph("<b>Recall</b>", body_style)
        ],
        [
            Paragraph("Model 1 Baseline", body_style),
            Paragraph("512×512", body_style),
            Paragraph("Dice + BCE", body_style),
            Paragraph("0.6990", body_style),
            Paragraph("0.5399", body_style),
            Paragraph("70.90%", body_style),
            Paragraph("69.89%", body_style)
        ],
        [
            Paragraph("Model 2 ResNet-34", body_style),
            Paragraph("512×512", body_style),
            Paragraph("Dice + BCE", body_style),
            Paragraph("0.7235", body_style),
            Paragraph("0.5694", body_style),
            Paragraph("72.01%", body_style),
            Paragraph("72.71%", body_style)
        ],
        [
            Paragraph("Model 3 Champion (512px)", body_style),
            Paragraph("512×512", body_style),
            Paragraph("Dice + Focal + Boundary", body_style),
            Paragraph("0.7249", body_style),
            Paragraph("0.5723", body_style),
            Paragraph("72.38%", body_style),
            Paragraph("73.51%", body_style)
        ],
        [
            Paragraph("Model 4 High-Recall (768px)", body_style),
            Paragraph("768×768", body_style),
            Paragraph("Dice + Focal + Boundary", body_style),
            Paragraph("0.7207", body_style),
            Paragraph("0.5708", body_style),
            Paragraph("70.57%", body_style),
            Paragraph("75.72%", body_style)
        ],
        [
            Paragraph("<b>Patch Refiner (Epoch 1)</b>", body_style),
            Paragraph("512×512 Native", body_style),
            Paragraph("Dice + Focal + Boundary", body_style),
            Paragraph("0.6740", body_style),
            Paragraph("0.5146", body_style),
            Paragraph("62.80%", body_style),
            Paragraph("76.66%", body_style)
        ],
        [
            Paragraph("<b>Patch Refiner (Epoch 2)</b>", body_style),
            Paragraph("512×512 Native", body_style),
            Paragraph("Dice + Focal + Boundary", body_style),
            Paragraph("0.7135", body_style),
            Paragraph("0.5609", body_style),
            Paragraph("72.29%", body_style),
            Paragraph("73.17%", body_style)
        ],
        [
            Paragraph("<b>Patch Refiner (Epoch 14 — SOTA)</b>", body_style),
            Paragraph("512×512 Native", body_style),
            Paragraph("Dice + Focal + Boundary", body_style),
            Paragraph("<b>0.7260</b>", body_style),
            Paragraph("<b>0.5756</b>", body_style),
            Paragraph("<b>69.01%</b>", body_style),
            Paragraph("<b>79.37%</b>", body_style)
        ]
    ]

    t_res = Table(results_data, colWidths=[110, 65, 110, 50, 50, 55, 55])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#DCFCE7')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 10))

    # Section 5: Operational Guide (Pause, Resume, Monitor)
    story.append(Paragraph("5. Operational Guide: How to Pause, Resume & Monitor Training", h1_style))
    story.append(Paragraph(
        "To manage training safely when leaving your computer, use the following operational procedures:",
        body_style
    ))

    story.append(Paragraph("<b>A. How to Check Live Training Progress / Results Table:</b>", h2_style))
    story.append(Paragraph("Run the live summary table command in terminal to view all completed epochs:", body_style))
    story.append(Paragraph("<code>python show_results.py</code>", code_style))
    story.append(Paragraph("To see the animated, dynamic progress bar with GPU telemetry:", body_style))
    story.append(Paragraph("<code>python live_progress.py</code>", code_style))

    story.append(Paragraph("<b>B. How to Pause / Stop the Training:</b>", h2_style))
    story.append(Paragraph(
        "1. Open the terminal where training is running and press <b>Ctrl + C</b>.<br/>"
        "2. The latest epoch metrics and model weights are automatically saved to <code>checkpoints/patch_refiner_latest.pth</code> and <code>checkpoints/patch_refiner_best.pth</code>.",
        body_style
    ))

    story.append(Paragraph("<b>C. How to Resume / Continue the Training:</b>", h2_style))
    story.append(Paragraph(
        "When you return and want to continue training from the exact epoch where you paused, run:",
        body_style
    ))
    story.append(Paragraph("<code>python training/train_patch_refiner.py --resume</code>", code_style))
    story.append(Paragraph(
        "The training script will automatically restore model weights, AdamW optimizer momentum, CosineAnnealing learning rate schedule, and AMP gradient scaler.",
        body_style
    ))

    story.append(Paragraph("<b>D. How to Launch the Web Dashboard:</b>", h2_style))
    story.append(Paragraph("<code>python dashboard/app.py</code>", code_style))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully generated PDF: {filename}")


if __name__ == '__main__':
    target_pdf = "reports/The_New_Improved_Model_Details.pdf"
    build_pdf(target_pdf)
    
    # Also save with spaced name as requested
    spaced_pdf = "reports/The New Improved Model Details.pdf"
    shutil.copyfile(target_pdf, spaced_pdf)
    print(f"[+] Copied to: {spaced_pdf}")
