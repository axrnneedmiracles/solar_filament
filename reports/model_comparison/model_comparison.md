# Solar Filament AI Research System: Scientific Model Comparison Report
**Report Date:** 2026-08-20  
**Dataset:** MAGFiLO 1.0 (Kaggle 2026) | Seed 42 Split  
**Evaluation Accelerator:** NVIDIA GeForce RTX 4050 GPU  

---

## 1. Master Model Comparison Matrix

| ID | Model Architecture | Resolution | Channels | Backbone | Loss Formulation | Best Epoch | Val Dice | Val IoU | Precision | Recall | Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1** | Baseline Mask2Former | 512×512 | 1 | Custom Light Conv | 0.50 Dice + 0.50 BCE | 46 | `0.6990` | `0.5399` | `0.7090` | `0.6989` | **COMPLETED** |
| **M2** | Pretrained ResNet-34 | 512×512 | 1 | ResNet-34 ImageNet | 0.50 Dice + 0.50 BCE | 45 | `0.7235` | `0.5695` | `0.7369` | `0.7183` | **COMPLETED** |
| **M3** | ResNet-34 + Hybrid Loss | 512×512 | 1 | ResNet-34 ImageNet | 0.40 Dice + 0.30 Focal + 0.30 Boundary | 49 | **`0.7249`** | **`0.5723`** | `0.7238` | `0.7351` | 🏆 **COMPLETED (512px Peak)** |
| **M4** | Heavy Augmentation | 512×512 | 1 | ResNet-34 ImageNet | 0.40 Dice + 0.30 Focal + 0.30 Boundary | 41 | `0.6971` | `0.5375` | `0.6802` | `0.7237` | **REJECTED** |
| **M5** | High-Res 768×768 | 768×768 | 1 | ResNet-34 ImageNet | 0.40 Dice + 0.30 Focal + 0.30 Boundary | 50 | `0.7207` | `0.5708` | `0.7057` | **`0.7572`** | 🏆 **COMPLETED (Recall Peak)** |
| **M6** | Frangi + Hessian (3-Ch) | 512×512 | 3 | ResNet-34 (3-Ch) | 0.40 Dice + 0.30 Focal + 0.30 Boundary | 21 | `0.4872` | `0.3346` | `0.5416` | `0.4618` | **IN PROGRESS (Epoch 31/50)** |

---

## 2. Detailed Progression & Hypothesis Validation

### Experiment 1: Baseline to Pretrained ResNet-34 (+2.45% Dice)
* **Hypothesis:** Transfer learning from ImageNet provides rich low-level ridge and boundary feature representations that outperform randomly initialized shallow conv layers.
* **Result:** **CONFIRMED.** Dice increased from `0.6990` to `0.7235`, and IoU improved from `0.5399` to `0.5695`.

### Experiment 2: BCE Loss to Tri-Component Hybrid Loss (+0.14% Dice, +1.68% Recall)
* **Hypothesis:** Combining Soft Dice ($40\%$), Focal Loss ($30\%$, $\alpha=0.75, \gamma=2.0$), and Morphological Boundary Loss ($30\%$) balances foreground class imbalance and prevents broken filament boundaries.
* **Result:** **CONFIRMED.** Model 3 achieved the highest overall 512px Dice (`0.7249`) and IoU (`0.5723`).

### Experiment 3: Heavy Astronomical Augmentation (-2.78% Dice — REJECTED)
* **Hypothesis:** Synthetic atmospheric seeing blur, Gaussian noise, and elastic distortion would improve generalization.
* **Result:** **REFUTED & REJECTED.** Dice dropped to `0.6971`. Solar filaments are sharp chromospheric magnetic flux ropes; synthetic blur destroyed sub-pixel boundary gradients.

### Experiment 4: Native 768×768 Spatial Resolution (+2.21% Recall — HIGH RECALL CHAMPION)
* **Hypothesis:** Increasing spatial resolution preserves thin filament barbs and fragmented filament spines that are downsampled out at 512px.
* **Result:** **CONFIRMED.** Model 5 achieved the all-time highest validation recall (**`75.72%`**), correctly segmenting faint threads missed at 512px.

### Experiment 5: 3-Channel Frangi + Hessian Input (IN PROGRESS — Lower Dice)
* **Hypothesis:** Providing classical Frangi vesselness and Hessian ridge responses as explicit input channels would accelerate spine segmentation.
* **Result:** **CURRENTLY LOWER.** At Epoch 31/50, peak validation Dice is `0.4872` (Epoch 21). Second-order Gaussian derivatives zero out diffuse filament edges, suppressing gradient backpropagation.

---

## 3. Champion Model Designation

* **Best Global Dice Model:** **Model 3** (`phase2_hybrid_loss_dice0.7249.pth` @ 512px)
* **Best Global Recall Model:** **Model 5** (`phase3_768res_dice0.7207.pth` @ 768px)
* **Current Production Deployment:** Both models are integrated into [`dashboard/app.py`](file:///c:/Users/aryan/OneDrive/Desktop/solarf/dashboard/app.py) with dynamic switching.
