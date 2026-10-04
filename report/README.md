# FairFake: Research Paper & Publication Reports (IEEE Format)

This directory contains the complete IEEE-formatted academic paper, formatted publication tables, camera-ready PDF, and LaTeX source files for the **FairFake (TruthLens)** deepfake detection and fairness auditing project.

---

## 📄 Generated Deliverables & Files Overview

1. **`FairFake_IEEE_Paper.pdf`** *(Camera-Ready Publication PDF)*:
   - Full, publication-quality **IEEE two-column conference paper**.
   - Includes formatted Title block, Author blocks, Abstract & Keywords, Equations (1)–(17), Algorithm 1, Tables I, II, and III, and IEEE References.
   - Ready to view, share, or submit directly.

2. **`paper.html`** *(Standalone Interactive & Printable Paper)*:
   - High-fidelity IEEE conference styling built with standard CSS column layout.
   - Includes full-width spanning table (`TABLE II`), single-column tables (`TABLE I` and `TABLE III`), boxed algorithmic framework, and print stylesheet.
   - Can be opened directly in any modern browser or printed/exported via browser print dialog (`Ctrl+P` / `Cmd+P`).

3. **`main.tex`** *(Verified LaTeX Source File)*:
   - Formatted strictly in **IEEE conference / publication style** (`IEEEtran`).
   - **Corrected Tables:**
     - **Table I:** Resized and padded to fit single-column boundaries (`\columnwidth = 3.5 in`) without margin overflow.
     - **Table II:** Spans two columns (`table*`) with all **10 declared columns** (`\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccccc}`), fixing the previous column mismatch error, with clean multi-row attribute categories, math-aligned signs, and color-coded severity badges.
     - **Table III:** Metric definitions and regulatory compliance action matrix.
     - **Algorithm 1:** Boxed demographic auditing pseudocode with balanced control group sampling.
   - Balanced columns and microtypography (`microtype`, `booktabs`, `flushend`).

4. **`references.bib`** *(BibTeX Database)*:
   - Complete BibTeX entries with peer-reviewed citations from IEEE, CVPR, ICCV, NeurIPS, and ICML.

---

## 📊 Summary of Formatted Tables in the Paper

### Table I: Overall Forensic Detection Performance Across Models ($N = 12,040$)
| Model Architecture | Acc (%) | FPR | FNR | F1-Score | AUC | High-Bias Attributes |
|:-------------------|:-------:|:---:|:---:|:--------:|:---:|:--------------------:|
| Xception [8] | 86.5% | 0.12 | 0.15 | 0.86 | 0.924 | 4 |
| **EfficientNet-B0 [9]** | **89.2%** | **0.09** | **0.11** | **0.89** | **0.948** | **2** |
| Dual-Stream (RGB+FFT) | 88.4% | 0.10 | 0.13 | 0.88 | 0.939 | 3 |

### Table II: Comprehensive Demographic and Attribute Fairness Audit ($N = 12,040$)
Spans **10 columns** across both paper columns:
- **Columns:** Category, Attribute Name, Model, $Err(\text{With})$, $Err(\text{W/O})$, $RP$, $CRP$, $PDRP$, $DDRP$, Severity.
- **Audited Categories:**
  - *Demographics & Skin Tone:* Dark Skin ($CRP = +0.55$, HIGH), Asian ($CRP = +0.04$, LOW).
  - *Age Cohorts:* Senior ($Age \ge 55$, $CRP = +0.38$, MODERATE), Young ($Age < 30$, $CRP = +0.02$, LOW).
  - *Gender:* Male ($CRP = +0.10$, LOW), Female ($CRP = +0.08$, LOW).
  - *Cosmetics & Accessories:* Eyeglasses ($CRP = +0.45$, HIGH), Heavy Makeup ($CRP = +0.48$, HIGH).
  - *Hair & Facial Hair:* Bald ($CRP = +0.52$, HIGH), Facial Hair ($CRP = +0.22$, MODERATE), Blond Hair ($CRP = +0.15$, LOW).
  - *Facial Structure & Expression:* Chubby / Round Face ($CRP = +0.19$, LOW), Smiling ($CRP = +0.03$, LOW).

### Table III: Fairness Metric Taxonomy & Regulatory Thresholds
Provides formal mathematical definitions and action thresholds for $RP(a)$, $CRP(a)$, $PDRP(a)$, and $DDRP(a)$ under the EU AI Act and NIST AI RMF.

---

## 🚀 How to Recompile LaTeX (Overleaf or Local)

### Option 1: Overleaf (Zero Setup)
1. Go to [Overleaf](https://www.overleaf.com).
2. Create a **New Project** $\rightarrow$ **Upload Project**.
3. Upload `main.tex` and `references.bib`.
4. Click **Recompile**. The document will compile with zero errors into IEEE double-column layout.

### Option 2: Local Compilation
```bash
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
Or with `latexmk`:
```bash
latexmk -pdf main.tex
```
