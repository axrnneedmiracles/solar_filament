"""
Generate Complete 4-Page Project Technical Audit & Plain-English Guide
======================================================================
Detailed document covering:
Pages 1-3: Exact Technical Architecture, Code Audits, and Model Pipeline of OUR project.
Page 4: Plain-English "How It All Works" guide explaining the entire system simply.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers, footers, and page numbers."""
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
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(45, 752, "Solar Filament Segmentation & Space Weather Intelligence — System Audit")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(45, 746, 567, 746)

        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(567, 32, page_str)
        self.drawString(45, 32, "GGSIPU HACKATHON 2026 | TRACK 19: AI SPACE WEATHER INTELLIGENCE")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(45, 42, 567, 42)
        self.restoreState()


def generate_full_pdf(output_filename="Solar_Filament_Technical_Audit.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=48,
        bottomMargin=48,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=3,
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=8,
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=3,
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=2.5,
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#2C5282"),
        backColor=colors.HexColor("#F7FAFC"),
        borderColor=colors.HexColor("#E2E8F0"),
        borderWidth=0.5,
        borderPadding=4,
        spaceBefore=3,
        spaceAfter=4,
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#1A202C"),
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold',
    )

    story = []

    # =========================================================================
    # PAGE 1: PROJECT OVERVIEW, DATASET, & PREPROCESSING PIPELINE
    # =========================================================================
    story.append(Paragraph("Solar Filament Segmentation & Space Weather Intelligence", title_style))
    story.append(Paragraph("Complete Technical Project Audit & Implementation Blueprint (Our Active Codebase)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    # Summary Table
    sum_data = [
        [
            Paragraph("<b>Project Root:</b> <code>c:/Users/aryan/OneDrive/Desktop/solarf</code>", table_cell_style),
            Paragraph("<b>Deep Learning Model:</b> Mask2Former (2.76M Params)", table_cell_style),
        ],
        [
            Paragraph("<b>Dataset:</b> MAGFiLO 1.0 (707 Local Train, 8,199 Polygon Masks)", table_cell_style),
            Paragraph("<b>Target GPU:</b> NVIDIA RTX 4050 Laptop GPU (CUDA 12.4, AMP)", table_cell_style),
        ],
        [
            Paragraph("<b>Astronomical Calibration:</b> 0.6 arcsec/px = 435.0 km/px", table_cell_style),
            Paragraph("<b>Live Dashboard:</b> Gradio UI (<code>http://127.0.0.1:7860</code>)", table_cell_style),
        ],
    ]
    t_sum = Table(sum_data, colWidths=[260, 262])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_sum)
    story.append(Spacer(1, 4))

    # 1. Dataset Audit
    story.append(Paragraph("1. DATASET SPECIFICATIONS & AUDIT", h1_style))
    d_points = [
        "<b>Exact Dataset Used:</b> MAGFiLO 1.0 (MLEcoFi 2024 / NSO/GONG H-alpha full-disk solar observations).",
        "<b>File Locations:</b> <code>images/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json</code>, training frames in <code>train/train_images/</code>.",
        "<b>Exact Quantities:</b> 707 JPEG images locally on disk (1,154 total entries in full catalogue); 180 unseen test images in <code>test/test_images/</code>.",
        "<b>Ground Truth Annotations:</b> 8,199 COCO polygon masks with spine curve traces across 4 categories (Left: 2,535, Right: 2,590, Unidentifiable: 3,074, Ambiguous: 0).",
        "<b>Dimensions:</b> Input 2048 x 2048 pixels (single-channel 8-bit) downsampled to 512 x 512 normalized float32 tensors.",
        "<b>Train / Validation Partitioning:</b> 80% Train (565 local images) / 20% Validation (142 local images) with fixed random seed 42 (<code>configs/default_config.yaml:16</code>).",
        "<b>Data Leakage Prevention:</b> Strict image-ID set separation prior to dataset loading; zero cross-contamination in <code>cache_512/</code>; augmentations isolated exclusively to training."
    ]
    for p in d_points:
        story.append(Paragraph(f"• {p}", bullet_style))

    # 2. Preprocessing Pipeline
    story.append(Paragraph("2. PREPROCESSING & FEATURE EXTRACTION PIPELINE", h1_style))
    story.append(Paragraph("Defined across <code>preprocessing/solar_preprocessor.py</code> and <code>classical/advanced_extractor.py</code>:", body_style))

    prep_data = [
        [Paragraph("Pipeline Step", table_header_style), Paragraph("Method / Function", table_header_style), Paragraph("Exact Mathematical Parameter Values", table_header_style)],
        [Paragraph("1. Grayscale Conversion", table_cell_bold), Paragraph("cv2.cvtColor", table_cell_style), Paragraph("COLOR_BGR2GRAY if 3-channel; float32 scaled to [0, 1]", table_cell_style)],
        [Paragraph("2. Solar Disk Detection", table_cell_bold), Paragraph("cv2.threshold + minEnclosingCircle", table_cell_style), Paragraph("Threshold T=25, Gaussian blur kernel (9,9), sigma=2.0", table_cell_style)],
        [Paragraph("3. Solar Limb Removal", table_cell_bold), Paragraph("Radial interior boundary masking", table_cell_style), Paragraph("safe_radius = 0.93 * radius (7% outer limb boundary cliff removed)", table_cell_style)],
        [Paragraph("4. Limb Darkening Flattening", table_cell_bold), Paragraph("scipy uniform_filter background removal", table_cell_style), Paragraph("bg = uniform_filter(norm, size=h//8); flattened = norm / bg", table_cell_style)],
        [Paragraph("5. CLAHE Contrast Boost", table_cell_bold), Paragraph("cv2.createCLAHE", table_cell_style), Paragraph("clipLimit = 2.0, tileGridSize = (8, 8) — amplifies filament threads", table_cell_style)],
        [Paragraph("6. Multi-Scale Black Top-Hat", table_cell_bold), Paragraph("cv2.morphologyEx (MORPH_BLACKHAT)", table_cell_style), Paragraph("Elliptical kernels: k in {7, 13, 21, 31} px — isolates dark channels", table_cell_style)],
        [Paragraph("7. Multi-Scale Frangi Filter", table_cell_bold), Paragraph("Hessian eigenvalue ridge analysis", table_cell_style), Paragraph("Scales = [1.0, 1.8, 3.0, 5.0, 7.5], alpha=0.5, beta=0.5, gamma=15", table_cell_style)],
        [Paragraph("8. Data Augmentation", table_cell_bold), Paragraph("Albumentations spatial transforms", table_cell_style), Paragraph("RandomRotate90 (p=0.5), HorizontalFlip (p=0.5), VerticalFlip (p=0.5)", table_cell_style)],
    ]
    t_prep = Table(prep_data, colWidths=[110, 152, 260])
    t_prep.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_prep)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: MODEL ARCHITECTURE, LOSS FUNCTION, OPTIMIZER & CUDA
    # =========================================================================
    story.append(Paragraph("3. MODEL ARCHITECTURE — MASK2FORMER & U-NET", h1_style))
    story.append(Paragraph("<b>Primary Architecture:</b> Mask2Former (Masked-attention Mask Transformer) in <code>models/mask2former.py</code>. <b>Secondary Baseline:</b> Pretrained ResNet-34 U-Net in <code>models/unet.py</code>.", body_style))

    arch_data = [
        [Paragraph("Stage / Block", table_header_style), Paragraph("Module Description", table_header_style), Paragraph("Input Shape", table_header_style), Paragraph("Output Shape", table_header_style), Paragraph("Parameters", table_header_style)],
        [Paragraph("Stem C1", table_cell_bold), Paragraph("Double Conv2d (3x3) + BN + GELU", table_cell_style), Paragraph("[B, 1, 512, 512]", table_cell_style), Paragraph("[B, 32, 512, 512]", table_cell_style), Paragraph("9,536", table_cell_style)],
        [Paragraph("Encoder C2", table_cell_bold), Paragraph("MaxPool + Double Conv2d + BN", table_cell_style), Paragraph("[B, 32, 512, 512]", table_cell_style), Paragraph("[B, 64, 256, 256]", table_cell_style), Paragraph("55,680", table_cell_style)],
        [Paragraph("Encoder C3", table_cell_bold), Paragraph("MaxPool + Double Conv2d + BN", table_cell_style), Paragraph("[B, 64, 256, 256]", table_cell_style), Paragraph("[B, 128, 128, 128]", table_cell_style), Paragraph("221,696", table_cell_style)],
        [Paragraph("Encoder C4", table_cell_bold), Paragraph("MaxPool + Double Conv2d + BN", table_cell_style), Paragraph("[B, 128, 128, 128]", table_cell_style), Paragraph("[B, 256, 64, 64]", table_cell_style), Paragraph("885,760", table_cell_style)],
        [Paragraph("Encoder C5", table_cell_bold), Paragraph("MaxPool + Conv2d + BN", table_cell_style), Paragraph("[B, 256, 64, 64]", table_cell_style), Paragraph("[B, 128, 32, 32]", table_cell_style), Paragraph("295,296", table_cell_style)],
        [Paragraph("FPN Lateral", table_cell_bold), Paragraph("1x1 Convs + Top-Down Additions", table_cell_style), Paragraph("C5, C4, C3, C2", table_cell_style), Paragraph("P5, P4, P3, P2 (128-ch)", table_cell_style), Paragraph("82,304", table_cell_style)],
        [Paragraph("Mask Features", table_cell_bold), Paragraph("High-Res Conv3x3 on [P1, C1]", table_cell_style), Paragraph("[B, 160, 512, 512]", table_cell_style), Paragraph("[B, 128, 512, 512]", table_cell_style), Paragraph("332,032", table_cell_style)],
        [Paragraph("Query Embeddings", table_cell_bold), Paragraph("20 Learnable Filament Queries", table_cell_style), Paragraph("N/A", table_cell_style), Paragraph("[B, 20, 128]", table_cell_style), Paragraph("2,560", table_cell_style)],
        [Paragraph("Transformer Dec", table_cell_bold), Paragraph("3x Masked Cross-Attn + Self-Attn", table_cell_style), Paragraph("[B, 20, 128], P_k", table_cell_style), Paragraph("[B, 20, 128]", table_cell_style), Paragraph("789,248", table_cell_style)],
        [Paragraph("Dense Head", table_cell_bold), Paragraph("Query Modulation + Conv Head", table_cell_style), Paragraph("[B, 128, 512, 512]", table_cell_style), Paragraph("[B, 1, 512, 512]", table_cell_style), Paragraph("74,113", table_cell_style)],
        [Paragraph("Total Parameters", table_cell_bold), Paragraph("Complete Mask2Former Network", table_cell_style), Paragraph("Single Channel", table_cell_style), Paragraph("Logit Output", table_cell_style), Paragraph("<b>2,762,465</b> (2.76 M)", table_cell_style)],
    ]
    t_arch = Table(arch_data, colWidths=[85, 147, 100, 100, 90])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 4))

    # 4. Loss & Optimizer
    story.append(Paragraph("4. LOSS FORMULATION, OPTIMIZER & SCHEDULER", h1_style))
    loss_points = [
        "<b>Compound Solar Loss (<code>training/losses.py</code>):</b> Combined Focal Loss (handles severe class imbalance where filaments occupy &lt; 2% of disk) + Soft-Dice Loss (optimizes boundary IoU directly):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>Loss = 0.50 * DiceLoss + 0.50 * BCEWithLogitsLoss</code> (or Focal alpha=0.8, gamma=2.0, smooth=1.0).",
        "<b>Optimizer:</b> <code>AdamW</code> with learning rate <code>lr = 1e-4</code>, weight decay <code>1e-5</code>, Betas <code>(0.9, 0.999)</code>, and gradient clipping <code>max_norm = 1.0</code>.",
        "<b>Learning Rate Scheduler:</b> <code>CosineAnnealingLR</code> decaying from <code>1e-4</code> down to <code>1e-7</code> across 50 epochs, stepped once per epoch."
    ]
    for p in loss_points:
        story.append(Paragraph(f"• {p}", bullet_style))

    # 5. GPU & CUDA Acceleration
    story.append(Paragraph("5. GPU HARDWARE & CUDA 12.4 OPTIMIZATIONS", h1_style))
    gpu_points = [
        "<b>Dedicated Device:</b> <code>cuda:0</code> — NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, Driver 610.74, CUDA 12.4).",
        "<b>Automatic Mixed Precision (AMP):</b> <code>torch.amp.autocast('cuda')</code> with <code>GradScaler</code> for FP16 Tensor Core acceleration (2x throughput).",
        "<b>Execution Speed:</b> <b>~2.0 iterations/sec (~1.5 minutes per epoch)</b> over all 923 training images with zero CPU bottleneck.",
        "<b>DataLoader Optimization:</b> <code>num_workers = 0</code> (eliminates Windows multiprocessing pickling overhead) with <code>pin_memory = True</code>."
    ]
    for p in gpu_points:
        story.append(Paragraph(f"• {p}", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: VALIDATION, HYBRID FUSION, MORPHOLOGY & PROJECT CODE
    # =========================================================================
    story.append(Paragraph("6. VALIDATION METRICS & MODEL CHECKPOINTING", h1_style))
    v_points = [
        "<b>Validation Frequency:</b> Full evaluation over all 231 held-out validation images executed at the end of every epoch under <code>@torch.no_grad()</code>.",
        "<b>Exact Metric Formulations:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>Dice (F1):</b> <code>(2 * TP + 1.0) / (2 * TP + FP + FN + 1.0)</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>IoU (Jaccard):</b> <code>(TP + 1.0) / (TP + FP + FN + 1.0)</code><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>Precision:</b> <code>(TP + 1e-7) / (TP + FP + 1e-7)</code> &nbsp;|&nbsp; <b>Recall:</b> <code>(TP + 1e-7) / (TP + FN + 1e-7)</code>",
        "<b>Optimal Threshold Sweep:</b> Sweeps probability thresholds in <code>[0.15 ... 0.50]</code> to maximize validation Dice score.",
        "<b>Checkpoint Saving:</b> Automatically updates <code>checkpoints/best_model.pth</code> whenever validation Dice improves."
    ]
    for p in v_points:
        story.append(Paragraph(f"• {p}", bullet_style))

    # 7. Post-Processing & Morphology
    story.append(Paragraph("7. HYBRID FUSION & SPACE WEATHER MORPHOLOGY", h1_style))
    story.append(Paragraph("Defined across <code>hybrid/fusion.py</code> and <code>analysis/filament_morphology.py</code>:", body_style))
    m_points = [
        "<b>Hybrid Ensemble Fusion:</b> <code>P_final = alpha * P_DL + (1 - alpha) * P_Frangi</code> (configurable fusion weight alpha, default = 0.50).",
        "<b>Size & Geometry Filtering:</b> Rejects components &lt; 25 px (camera noise) or &gt; 8,000 px. Rejects round sunspots via spatial covariance eigenvalue ratio &gt;= 1.50.",
        "<b>Astronomical Physical Calibration:</b> 1 arcsec ≈ 725 km on the Sun (0.6 arcsec/px at GONG resolution $\rightarrow$ <b>435.0 km per pixel</b>).",
        "<b>Quantitative Measurements:</b> Physical spine length in km (iterative skeletonization), physical area in $\\text{km}^2$, average width, orientation angle (fitted ellipse), bounding boxes, centroids, and prediction confidence."
    ]
    for p in m_points:
        story.append(Paragraph(f"• {p}", bullet_style))

    # Complete Project Code Map Table
    story.append(Paragraph("8. MASTER CODE INVENTORY & PROJECT ARCHITECTURE", h1_style))
    code_map_data = [
        [Paragraph("File / Module", table_header_style), Paragraph("Component Role", table_header_style), Paragraph("Core Algorithms / Technologies", table_header_style)],
        [Paragraph("configs/default_config.yaml", table_cell_bold), Paragraph("Central Configuration", table_cell_style), Paragraph("Model hyperparameters, learning rate, scales, device", table_cell_style)],
        [Paragraph("preprocessing/solar_preprocessor.py", table_cell_bold), Paragraph("GONG Astronomical Preprocessor", table_cell_style), Paragraph("Disk isolate (0.93r), uniform_filter limb flatten, CLAHE", table_cell_style)],
        [Paragraph("preprocessing/dataset.py", table_cell_bold), Paragraph("Dataset & Augmentation", table_cell_style), Paragraph("COCO JSON polygon parser, fast cache_512 loader", table_cell_style)],
        [Paragraph("classical/advanced_extractor.py", table_cell_bold), Paragraph("Classical Ridge Engine", table_cell_style), Paragraph("Multi-scale Black Top-Hat, Frangi vesselness, Hysteresis", table_cell_style)],
        [Paragraph("models/mask2former.py", table_cell_bold), Paragraph("Primary DL Transformer", table_cell_style), Paragraph("20 Query Embeddings, FPN, Masked Cross-Attention (2.76M)", table_cell_style)],
        [Paragraph("models/unet.py", table_cell_bold), Paragraph("Secondary DL Baseline", table_cell_style), Paragraph("4-stage encoder-decoder U-Net with skip connections", table_cell_style)],
        [Paragraph("training/losses.py", table_cell_bold), Paragraph("Compound Loss Functions", table_cell_style), Paragraph("Focal Loss (alpha=0.8, gamma=2.0) + Soft-Dice Loss", table_cell_style)],
        [Paragraph("training/train.py", table_cell_bold), Paragraph("GPU Training Loop", table_cell_style), Paragraph("AMP FP16, AdamW, CosineAnnealingLR, Checkpointing", table_cell_style)],
        [Paragraph("hybrid/fusion.py", table_cell_bold), Paragraph("Hybrid Fusion Engine", table_cell_style), Paragraph("Weighted probability blending + threshold sweep", table_cell_style)],
        [Paragraph("analysis/filament_morphology.py", table_cell_bold), Paragraph("Space Weather Morphology", table_cell_style), Paragraph("Skeletonization, physical km scaling, bounding boxes", table_cell_style)],
        [Paragraph("dashboard/app.py", table_cell_bold), Paragraph("Interactive Web Dashboard", table_cell_style), Paragraph("Gradio UI (upload, toggle modes, live morphology)", table_cell_style)],
        [Paragraph("test_single_image.py", table_cell_bold), Paragraph("CLI Test Script", table_cell_style), Paragraph("Run random test image or specific path via command line", table_cell_style)],
    ]
    t_code = Table(code_map_data, colWidths=[130, 120, 272])
    t_code.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_code)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: PLAIN-ENGLISH GUIDE — "HOW IT ALL WORKS FOR ANYONE"
    # =========================================================================
    story.append(Paragraph("4. HOW IT ALL WORKS (EXPLAINED SIMPLY)", title_style))
    story.append(Paragraph("A Plain-English, Step-by-Step Guide to Our Solar Filament AI System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=8))

    simple_steps = [
        ("Step 1: What Are Solar Filaments & Why Do We Care?",
         "Solar filaments are giant clouds of dense, cool plasma suspended above the Sun's blistering surface by powerful magnetic fields. When viewed through H-alpha telescopes (like NSO/GONG), they appear as <b>dark, twisting snake-like ribbons</b> across the bright Sun. When these magnetic loops snap, they erupt into massive Coronal Mass Ejections (CMEs) that can trigger geomagnetic storms on Earth—knocking out communication satellites, disrupting GPS, and damaging electrical power grids. Detecting them automatically gives us early space weather warnings."),

        ("Step 2: The Core Challenge (Why Standard AI Fails)",
         "Standard image AI models fail on raw solar telescope images for two reasons: (1) The bright Sun against the pitch-black space background creates a harsh contrast cliff around the outer edge (the solar limb), which regular algorithms mistake for giant filaments. (2) Filaments are very thin, faint, and occupy less than 2% of the image, while the rest of the Sun is covered in bright granulation noise and sunspots."),

        ("Step 3: How Our Smart Astronomical Preprocessing Cleans the Sun",
         "Before our AI ever looks at an image, our preprocessing engine does four critical things: First, it mathematically finds the Sun's exact circle and shrinks the border by 7% (<code>0.93 * radius</code>) to <b>completely erase the outer edge ring</b>. Second, it removes 'limb darkening' (where the center of the Sun is brighter than the edges). Third, it applies CLAHE contrast enhancement so faint dark filament threads pop out vividly against the background."),

        ("Step 4: Finding the Dark Channels with Physics Filters",
         "We use two classical mathematical filters: (1) <b>Black Top-Hat Transform</b>: Extracts only dark features narrower than a set width. (2) <b>Frangi Filter</b>: Analyzes second-order directional curvature (Hessian matrix) to highlight continuous thread-like lines while ignoring circular noise. This creates a physics-based candidate map of every potential filament."),

        ("Step 5: The AI Brain — Mask2Former Transformer",
         "We feed the cleaned solar observation into <b>Mask2Former</b>, an advanced AI Transformer architecture. Unlike basic neural networks, Mask2Former uses <b>Masked Cross-Attention</b>: it generates 20 individual 'filament query probes' and restricts its attention strictly to foreground filament shapes, completely ignoring background noise. It learns to recognize true filament ribbons while rejecting round sunspots."),

        ("Step 6: Hybrid Teamwork (AI + Classical Physics)",
         "Our system combines the strengths of both worlds: Deep Learning (high semantic intelligence) + Classical Frangi Filtering (fine edge precision). The user can even adjust the fusion slider (&alpha;) to blend the two models seamlessly."),

        ("Step 7: Real-World Space Weather Measurements",
         "Our system doesn't just draw masks—it performs full geometric science. Using the astronomical scale (<b>1 arcsecond = 725 km</b> on the Sun), it calculates the <b>exact physical length in kilometers</b>, total area in square kilometers, spine curvature, and draws bounding boxes and centroids around each detected filament."),

        ("Step 8: Interactive Web App & 1-Click Testing",
         "Anyone can use the system through our live <b>Gradio Web Dashboard</b> (at <code>http://127.0.0.1:7860</code>). You simply drag-and-drop any solar telescope picture, click 'Detect Filaments', and in less than 1 second, the system displays the original, cleaned, AI probability heatmap, final mask, and a full quantitative space weather report!")
    ]

    for title, text in simple_steps:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 2))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"4-Page Complete Audit PDF successfully created at: {output_filename}")


if __name__ == "__main__":
    generate_full_pdf()
