# Partial Arc Ellipse Fitting via Cloud Regression

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Journal: Pattern Recognition Letters](https://img.shields.io/badge/Journal-Pattern%20Recognition%20Letters-green.svg)](https://www.sciencedirect.com/journal/pattern-recognition-letters)

Official code for the paper:

> **Wansouwé, W.** (2026). *Partial Arc Ellipse Fitting via Cloud Regression: Critical Coverage Threshold and Sensitivity to Noise and Eccentricity.* Pattern Recognition Letters, Elsevier.

---

##  Abstract

This paper establishes that approximately **180° of arc coverage** constitutes a critical threshold for reliable ellipse parameter recovery within Granville's (2022) Cloud Regression framework. We provide:

1. **Empirical evidence** of the 180° critical threshold
2. **Spectral justification** via eigenvalue analysis of W^TW (λ₂ > 1 criterion)
3. **Power-law model** for the noise-threshold relationship: θ_c(σ) = −28.68° + 272.65° × σ^0.156

---

## Repository Structure

```
cloud-regression-ellipse/
├── fit_ellipse.py      ← Core ellipse fitting functions
├── experiments.py      ← Reproduce all paper figures
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/Wanbitching/cloud-regression-ellipse.git
cd cloud-regression-ellipse
pip install -r requirements.txt
```

### Run a quick demo

```bash
python fit_ellipse.py
```

### Reproduce all paper figures

```bash
python experiments.py --all
```

### Run a specific experiment

```bash
python experiments.py --exp 1   # Fig 2: baseline threshold
python experiments.py --exp 2   # Fig 3: eccentricity effect
python experiments.py --exp 3   # Fig 4: noise effect
python experiments.py --exp 4   # Fig 6: spectral analysis
python experiments.py --exp 5   # Fig 5: power-law model
```

---

## Method Overview

The Cloud Regression framework (Granville, 2022) reformulates ellipse fitting as:

```
Minimize:   θᵀ (WᵀW) θ
Subject to: θᵀθ = 1
```

where W contains artificial variables (x², xy, y², x, y, 1).
The solution θ* is the **eigenvector of WᵀW associated with λ_min**.

### Key Finding: Critical 180° Threshold

| Arc Coverage | λ₂ of WᵀW | Reliability |
|---|---|---|
| < 170° | < 1.0 |  Unreliable |
| ~ 180° | ≈ 1.0–2.0 |  Threshold |
| > 180° | > 1.0 |  Reliable |

### Power-Law Noise Model

```
θ_c(σ) = -28.68° + 272.65° × σ^0.156   (RSS = 79.7)
```

---

##  Main Results

| Condition | Critical Threshold |
|---|---|
| Low noise (σ ≤ 0.2) | ~180° |
| High eccentricity (e = 0.98) | ~160° |
| High noise (σ = 0.8) | ~260° |

---

##  Citation

If you use this code in your research, please cite:

```bibtex
@article{wansouwe2026ellipse,
  title   = {Partial Arc Ellipse Fitting via Cloud Regression:
             Critical Coverage Threshold and Sensitivity
             to Noise and Eccentricity},
  author  = {Wansouw{\'e}, Wanbitching},
  journal = {Pattern Recognition Letters},
  year    = {2026},
  publisher = {Elsevier}
}
```

Please also cite the original Cloud Regression paper:

```bibtex
@techreport{granville2022,
  title  = {Machine Learning Cloud Regression:
            The Swiss Army Knife of Optimization},
  author = {Granville, Vincent},
  year   = {2022},
  institution = {MLTechniques.com},
  note   = {Version 1.0}
}
```

---

##  License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

##  Contact

**Wansouwé Wanbitching**
Email: ericwansouwe@gmail.com
GitHub: [@Wanbitching](https://github.com/Wanbitching)

