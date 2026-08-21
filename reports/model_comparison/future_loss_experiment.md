# Controlled Protocol: Future Focal-Dice Loss (FDL) Experiment
**Protocol Status:** Prepared & Ready for Execution  

---

## 1. Current Champion Loss vs. Proposed FDL Formulation

### Current Champion Loss (Model 3):
$$\mathcal{L}_{\text{Hybrid}} = 0.40 \cdot \mathcal{L}_{\text{Dice}} + 0.30 \cdot \mathcal{L}_{\text{Focal}}(\alpha=0.75, \gamma=2.0) + 0.30 \cdot \mathcal{L}_{\text{Boundary}}$$

### Proposed Focal-Dice Loss (FDL) Formulation:
$$\mathcal{L}_{\text{FDL}} = \left(1 - \text{Dice}\right)^{\gamma_{\text{dice}}} + \lambda \cdot \mathcal{L}_{\text{Boundary}}$$
* **Mechanism:** Dynamically exponentially scales penalties when Dice is low ($<0.50$), forcing gradients to focus heavily on challenging low-contrast and fragmented filaments.

---

## 2. Controlled Experimental Controls

To guarantee scientific validity, the following parameters remain strictly locked:
* **Dataset:** MAGFiLO 1.0 (707 images)
* **Split:** Exact $80\% / 20\%$ split (Seed 42)
* **Resolution:** $512 \times 512\text{ px}$
* **Backbone:** ResNet-34 (ImageNet pretrained)
* **Optimizer:** AdamW $(\eta=10^{-4}, \text{weight decay}=10^{-5})$
* **Schedule:** Cosine Annealing over 50 epochs

---

## 3. Acceptance / Rejection Criteria

* **Acceptance:** Validation $\text{Dice} > 0.7249$ AND Validation $\text{Recall} > 73.51\%$.
* **Rejection:** Validation $\text{Dice} \le 0.7249$.
