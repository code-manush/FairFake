# FairFake: Research Paper & Project Report (IEEE Format)

This directory contains the complete IEEE-formatted academic paper cum project report for the **FairFake (TruthLens)** deepfake detection and fairness auditing project.

---

## 📄 Files Overview

- **`main.tex`**: The primary LaTeX document formatted strictly in **IEEE conference / publication style** (`IEEEtran`). It includes:
  - Complete mathematical formulations for **Relative Performance (RP)**, **Controlled Relative Performance (CRP)**, **Pristine Data Relative Performance (PDRP)**, and **Deepfake Data Relative Performance (DDRP)**.
  - Detailed system architecture spanning the **Next.js UI**, **FastAPI inference engine**, **Grad-CAM visual attribution**, and **DeepFace demographic parsing**.
  - Pseudocode for Algorithm 1: Demographic Fairness Audit with Control Group Sampling.
  - Full quantitative results matching the experimental benchmarks on the A-Celeb-DF / MAAD-Face dataset ($N = 12,040$).
  - Two IEEE-standard tables: Overall Model Performance and Comprehensive Attribute Fairness Audit across 34 attributes.
  - Ethical considerations, regulatory compliance (EU AI Act / NIST AI RMF), and algorithmic mitigation strategies.
- **`references.bib`**: Complete BibTeX database containing peer-reviewed citations (IEEE, CVPR, ICCV, NeurIPS, ICML).

---

## 🚀 How to Compile to PDF

### Option 1: Overleaf (Recommended - Fast & Zero Setup)
1. Go to [Overleaf](https://www.overleaf.com).
2. Click **New Project** $\rightarrow$ **Upload Project**.
3. Zip or upload the contents of this `report/` folder (`main.tex` and `references.bib`).
4. Click **Recompile** in Overleaf. The PDF will be generated immediately with IEEE double-column styling.

### Option 2: Local Compilation (pdflatex & bibtex)
If you have TeX Live, MacTeX, or MiKTeX installed:
```bash
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
Or simply run with `latexmk`:
```bash
latexmk -pdf main.tex
```

---

## ✏️ Customizing for Submission
Open `main.tex` and update the title block (lines 35–48) with your details:
- **Student Names**: Replace `Student Researcher(s)` and emails.
- **College / University**: Replace `Academic Institute / University`.
- **Guide / Professor**: Replace `Faculty Advisor / Project Mentor` with your professor's name and designation.
