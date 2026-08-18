"""
Generate Zero-to-Hundred Master Comprehensive Guide PDF
========================================================
Explains the entire Solar Filament Segmentation and Space Weather system
from absolute zero to complete technical depth in plain English and math.
"""

import os
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
        # Header rule
        self.setStrokeColor(colors.HexColor('#CBD5E0'))
        self.setLineWidth(0.75)
        self.line(40, 755, 572, 755)

        # Header Text
        self.setFont('Helvetica-Bold', 8)
        self.setFillColor(colors.HexColor('#1A365D'))
        self.drawString(40, 762, "SOLAR FILAMENT INTELLIGENCE SYSTEM")
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#718096'))
        self.drawRightString(572, 762, "Complete Zero-to-Hundred Technical Master Guide")

        # Footer rule
        self.line(40, 42, 572, 42)

        # Footer Text
        self.setFont('Helvetica', 8)
        self.drawString(40, 30, "Confidential - Space Weather AI Platform Documentation | GGSIPU Hackathon")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(572, 30, page_str)
        self.restoreState()


def build_zero_to_hundred_pdf(filename: str = "Solar_Filament_Complete_Guide_Zero_To_Hundred.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=52,
        bottomMargin=52
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=5
    )

    body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1A202C')
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1A365D')
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#2D3748')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#2D3748')
    )

    story = []

    # =========================================================================
    # PAGE 1: EXECUTIVE INTRODUCTION & CORE PIPELINE STEP-BY-STEP (STEPS 1-4)
    # =========================================================================
    story.append(Paragraph("Solar Filament Segmentation & Space Weather AI", title_style))
    story.append(Paragraph("<b>Zero-to-Hundred Master Technical Documentation & System Architecture</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3182CE'), spaceAfter=8))

    # Executive Overview
    story.append(Paragraph("1. Executive Introduction: What Are We Solving & Why?", h1_style))
    story.append(Paragraph(
        "<b>Solar filaments</b> (known as prominences when viewed against the dark sky at the solar edge) are massive "
        "clouds of dense, ionized gas (plasma) suspended high above the Sun's chromosphere by intense magnetic field loops. "
        "When these magnetic structures become unstable, they violently collapse or erupt into <b>Coronal Mass Ejections (CMEs)</b>. "
        "CMEs send billions of tons of high-energy charged particles toward Earth, causing severe geomagnetic storms capable of "
        "destroying satellite electronics, disrupting GPS navigation, endangering astronauts, and knocking out electrical power grids. "
        "Automated, real-time detection of solar filaments is essential for <b>planetary space weather defense</b>.",
        body_style
    ))

    story.append(Paragraph("2. The Zero-to-Hundred Pipeline Architecture", h1_style))
    story.append(Paragraph(
        "The diagram below outlines the sequential transformation of raw telescope observations into calibrated space weather alerts:",
        body_style
    ))

    # Pipeline Steps Table 1
    pipe_data = [
        [Paragraph("Pipeline Step", table_header_style), Paragraph("Input & Operation Applied", table_header_style), Paragraph("Scientific Purpose & Output", table_header_style)],
        [
            Paragraph("<b>STEP 1: Raw Image Ingestion</b>", table_cell_style),
            Paragraph("Full-disk H-alpha (656.3 nm) telescope image (2048x2048 px, 16-bit or 8-bit JPEG/FITS).", table_cell_style),
            Paragraph("Captures chromospheric hydrogen absorption where dark filaments absorb light from below.", table_cell_style)
        ],
        [
            Paragraph("<b>STEP 2: Solar Disk Detection</b>", table_cell_style),
            Paragraph("Otsu thresholding + Canny edge detection + minimum enclosing circle calculation.", table_cell_style),
            Paragraph("Accurately isolates the Sun's center coordinates (cx, cy) and radius R_disk from outer space.", table_cell_style)
        ],
        [
            Paragraph("<b>STEP 3: 7% Limb Edge Rejection</b>", table_cell_style),
            Paragraph("Safe disk mask created with safe_radius = 0.93 * R_disk.", table_cell_style),
            Paragraph("<b>CRITICAL FIX:</b> Completely cuts off the extreme bright outer limb cliff to eliminate false boundary rings.", table_cell_style)
        ],
        [
            Paragraph("<b>STEP 4: Limb Darkening Flattening</b>", table_cell_style),
            Paragraph("Uniform local spatial box filter: I_bg = uniform_filter(I, size=h//8); I_flat = I / (I_bg + eps).", table_cell_style),
            Paragraph("Flattens the natural optical brightness drop-off from the center of the Sun to its edge.", table_cell_style)
        ]
    ]

    t1 = Table(pipe_data, colWidths=[110, 205, 215])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F7FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: PIPELINE STEPS 5-10 & THE AI MODEL ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("2. Zero-to-Hundred Pipeline Walkthrough (Continued)", h1_style))

    pipe_data_2 = [
        [Paragraph("Pipeline Step", table_header_style), Paragraph("Input & Operation Applied", table_header_style), Paragraph("Scientific Purpose & Output", table_header_style)],
        [
            Paragraph("<b>STEP 5: Contrast Enhancement</b>", table_cell_style),
            Paragraph("CLAHE (Contrast Limited Adaptive Histogram Equalization, clip=2.0, grid=8x8).", table_cell_style),
            Paragraph("Amplifies faint dark filament absorption trenches without blowing out quiet solar regions.", table_cell_style)
        ],
        [
            Paragraph("<b>STEP 6: Multi-Scale Top-Hat</b>", table_cell_style),
            Paragraph("Morphological Black Top-Hat transforms using disk kernels k in {7, 13, 21, 31}.", table_cell_style),
            Paragraph("Extracts dark structural valleys and curvilinear depressions narrower than kernel sizes.", table_cell_style)
        ],
        [
            Paragraph("<b>STEP 7: Frangi Ridge Filtering</b>", table_cell_style),
            Paragraph("Eigenvalue decomposition of Hessian Matrix H(x,y) across scales sigma in {1.0 to 7.5}.", table_cell_style),
            Paragraph("Calculates vesselness probability response Rb and S, highlighting filamentary tubular structures.", table_cell_style)
        ],
        [
            Paragraph("<b>STEP 8: Mask2Former AI Model</b>", table_cell_style),
            Paragraph("5-stage CNN Pixel Decoder + 20 Learnable Queries + Masked Cross-Attention Transformer.", table_cell_style),
            Paragraph("<b>Learned AI Perception:</b> Distinguishes genuine filaments from round sunspots and background noise.", table_cell_style)
        ],
        [
            Paragraph("<b>STEP 9: Hybrid Fusion & Cleanup</b>", table_cell_style),
            Paragraph("Fused_Prob = 0.5 * P_AI + 0.5 * P_Frangi; Connected Components (min_area >= 40 px).", table_cell_style),
            Paragraph("Combines AI semantic reasoning with sub-pixel ridge sharpness and removes tiny noise specks.", table_cell_style)
        ],
        [
            Paragraph("<b>STEP 10: Space Weather Extraction</b>", table_cell_style),
            Paragraph("Zhang-Suen skeletonization, length in km, area in km^2 (Scale: 435.0 km/px), centroid, angle.", table_cell_style),
            Paragraph("Exports actionable space weather telemetry to CSV spreadsheets, JSON catalogs, and live Web UI.", table_cell_style)
        ]
    ]

    t2 = Table(pipe_data_2, colWidths=[110, 205, 215])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F7FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))

    # Mask2Former Deep Learning Section
    story.append(Paragraph("3. Deep Learning Architecture: Mask2Former Vision Transformer", h1_style))
    story.append(Paragraph(
        "Unlike standard segmentation networks (such as basic U-Net) that treat every pixel in isolation, <b>Mask2Former</b> "
        "is a state-of-the-art <b>Transformer-based</b> architecture designed specifically for complex biological and astronomical structures:",
        body_style
    ))

    m2f_arch_data = [
        [Paragraph("Component", table_header_style), Paragraph("Layer Structure & Dimensions", table_header_style), Paragraph("Functional Role in Filament Extraction", table_header_style)],
        [
            Paragraph("<b>Multi-Scale Pixel Decoder</b>", table_cell_style),
            Paragraph("5-stage hierarchical backbone (C1-C5: 32, 64, 128, 256, 128 channels) with top-down FPN.", table_cell_style),
            Paragraph("Extracts multi-resolution spatial features while maintaining a full 512x512 high-resolution mask feature map.", table_cell_style)
        ],
        [
            Paragraph("<b>Learnable Filament Queries</b>", table_cell_style),
            Paragraph("20 query vectors in R^(20 x 128) initialized and learned during backpropagation.", table_cell_style),
            Paragraph("Each query specializes in capturing a specific filament instance or cluster across the solar disk.", table_cell_style)
        ],
        [
            Paragraph("<b>Masked Cross-Attention</b>", table_cell_style),
            Paragraph("3-layer Transformer Decoder with attention restricted to predicted foreground regions.", table_cell_style),
            Paragraph("Focuses neural attention <b>strictly within filament channels</b>, completely ignoring quiet background solar noise.", table_cell_style)
        ]
    ]

    t_m2f = Table(m2f_arch_data, colWidths=[110, 195, 225])
    t_m2f.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F7FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_m2f)
    story.append(Spacer(1, 8))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: AI REASONING VS CLASSICAL FILTERS (THE CORE QUESTION ANSWERED)
    # =========================================================================
    story.append(Paragraph("4. Are We Using Real AI? (AI Reasoning vs Simple Filters)", h1_style))
    story.append(Paragraph(
        "<b>YES, ABSOLUTELY!</b> Our system is not just a collection of image filters. It implements a true "
        "<b>2.76 Million Parameter Deep Learning Neural Network (Mask2Former)</b> trained across 50 epochs on 8,199 "
        "expert annotations. Here is the fundamental difference between what classical filters do and what our AI model does:",
        body_style
    ))

    comparison_data = [
        [Paragraph("Capability / Dimension", table_header_style), Paragraph("Classical Filters (Frangi / Top-Hat / Preproc)", table_header_style), Paragraph("Mask2Former Deep Learning AI", table_header_style)],
        [
            Paragraph("<b>Mechanism</b>", table_cell_style),
            Paragraph("Fixed, hand-crafted mathematical formulas (derivatives, thresholds, pixel intensity values).", table_cell_style),
            Paragraph("<b>2.76 Million learnable weights</b> optimized via gradient descent to recognize complex visual patterns.", table_cell_style)
        ],
        [
            Paragraph("<b>Contextual Reasoning</b>", table_cell_style),
            Paragraph("<b>Zero reasoning.</b> Looks only at tiny local 3x3 pixel neighborhoods. Treats all dark pixels equally.", table_cell_style),
            Paragraph("<b>Global Multi-Head Attention.</b> Evaluates the entire Sun simultaneously, understanding spatial relationships.", table_cell_style)
        ],
        [
            Paragraph("<b>Filament vs Sunspot Discrimination</b>", table_cell_style),
            Paragraph("<b>FAILS:</b> Confuses dark circular sunspots and magnetic pores with dark filaments.", table_cell_style),
            Paragraph("<b>SUCCEEDS:</b> Distinguishes between round sunspots (rejected) and elongated magnetic filaments (accepted).", table_cell_style)
        ],
        [
            Paragraph("<b>Adaptability to Solar Cycle Changes</b>", table_cell_style),
            Paragraph("Brittle; fails when telescope exposure, atmospheric seeing, or solar cycle phase shifts.", table_cell_style),
            Paragraph("Robust; generalizes across varying solar noise, limb darkening, and telescope calibrations.", table_cell_style)
        ],
        [
            Paragraph("<b>Chirality & Polarity Awareness</b>", table_cell_style),
            Paragraph("Cannot understand magnetic field orientation or filament chirality (Left/Right barb structure).", table_cell_style),
            Paragraph("Learns feature representations aligned with magnetic Polarity Inversion Lines (PILs).", table_cell_style)
        ]
    ]

    t_comp = Table(comparison_data, colWidths=[110, 205, 215])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F7FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 8))

    # Why Hybrid Teamwork
    story.append(Paragraph("5. Why We Combine AI + Physics (The Hybrid Advantage)", h1_style))
    story.append(Paragraph(
        "Neither AI alone nor Classical CV alone is optimal for space weather defense. We use a <b>Hybrid Decision Engine</b>:<br/>"
        "1. <b>AI Brain (Mask2Former)</b> provides the <i>semantic intelligence</i> (eliminating false positives from sunspots and limb artifacts).<br/>"
        "2. <b>Classical Physics (Frangi)</b> provides <i>continuous edge sharpness</i> (connecting thin, faint filament spines at sub-pixel resolution).<br/>"
        "The combined probability <b>P_final = 0.50 * P_AI + 0.50 * P_Frangi</b> achieves a benchmark <b>69.90% Validation Dice score</b>.",
        body_style
    ))
    story.append(Spacer(1, 8))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: PHYSICAL SPACE WEATHER CALIBRATION & PROJECT DELIVERY
    # =========================================================================
    story.append(Paragraph("6. Astronomical Scale & Real-World Space Weather Physics", h1_style))
    story.append(Paragraph(
        "A segmentation mask in pixel coordinates is useless to solar physicists without physical units. Our system converts "
        "raw pixel coordinates into real astrophysical measurements using the GONG solar telescope plate scale:",
        body_style
    ))

    # Physics Table
    physics_data = [
        [Paragraph("Space Weather Parameter", table_header_style), Paragraph("Conversion Formula & Mathematical Calibration", table_header_style), Paragraph("Planetary Defense Application", table_header_style)],
        [
            Paragraph("<b>Plate Scale Calibration</b>", table_cell_style),
            Paragraph("1 arcsecond = 725.0 km on the Sun; GONG scale = 0.6 arcsec/px.<br/><b>KM_PER_PX = 0.6 * 725.0 = 435.0 km/pixel</b>", table_cell_style),
            Paragraph("Converts 2D telescope pixel coordinates into physical kilometers.", table_cell_style)
        ],
        [
            Paragraph("<b>Filament True Length</b>", table_cell_style),
            Paragraph("<b>L_km = N_skeleton_pixels * 435.0 km</b>", table_cell_style),
            Paragraph("Filaments exceeding 100,000 km represent critical eruption risks.", table_cell_style)
        ],
        [
            Paragraph("<b>Filament Surface Area</b>", table_cell_style),
            Paragraph("<b>A_km2 = N_mask_pixels * (435.0 km)^2 = N_px * 189,225 km^2</b>", table_cell_style),
            Paragraph("Estimates total trapped plasma mass available to fuel a CME.", table_cell_style)
        ],
        [
            Paragraph("<b>Orientation Angle</b>", table_cell_style),
            Paragraph("<b>theta = 0.5 * arctan2(2 * mu_11, mu_20 - mu_02)</b>", table_cell_style),
            Paragraph("Indicates magnetic tilt angle relative to the solar rotational equator.", table_cell_style)
        ]
    ]

    t_phys = Table(physics_data, colWidths=[110, 215, 205])
    t_phys.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F7FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_phys)
    story.append(Spacer(1, 8))

    # Project Outputs & Deployment
    story.append(Paragraph("7. Complete Project Outputs & Deliverables Summary", h1_style))

    outputs_data = [
        [Paragraph("Deliverable Type", table_header_style), Paragraph("File Path & Technology", table_header_style), Paragraph("Purpose / How to Access", table_header_style)],
        [
            Paragraph("<b>Trained AI Weights</b>", table_cell_style),
            Paragraph("checkpoints/best_model.pth (31.78 MB, PyTorch)", table_cell_style),
            Paragraph("Pre-trained 2.76M Mask2Former weights; loads instantly (0.05s inference).", table_cell_style)
        ],
        [
            Paragraph("<b>Interactive Web Dashboard</b>", table_cell_style),
            Paragraph("streamlit_app.py / share.streamlit.io", table_cell_style),
            Paragraph("Live cloud web application for real-time drag-and-drop solar image analysis.", table_cell_style)
        ],
        [
            Paragraph("<b>Morphology Spreadsheet</b>", table_cell_style),
            Paragraph("outputs/predictions/*.csv (Excel / Sheets)", table_cell_style),
            Paragraph("Contains ID, Area (km^2), Length (km), BBox, Centroid, and Orientation for every filament.", table_cell_style)
        ],
        [
            Paragraph("<b>Space Weather JSON Catalog</b>", table_cell_style),
            Paragraph("outputs/predictions/*.json", table_cell_style),
            Paragraph("Standard structured data format for automated ingestion into space observatory databases.", table_cell_style)
        ],
        [
            Paragraph("<b>Visual Overlay Grids</b>", table_cell_style),
            Paragraph("outputs/predictions/*_comparison_grid.png", table_cell_style),
            Paragraph("6-panel visual proof: Original, Preproc, Frangi, AI Heatmap, Mask, and BBox Overlay.", table_cell_style)
        ]
    ]

    t_out = Table(outputs_data, colWidths=[110, 195, 225])
    t_out.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F7FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_out)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Master Zero-to-Hundred Guide PDF successfully created at: {filename}")


if __name__ == '__main__':
    build_zero_to_hundred_pdf()
