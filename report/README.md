# FairFake: Research Paper & Auditing Publication Reports (IEEE Transactions Format)

This directory contains the complete IEEE Transactions-formatted academic paper, formatted publication tables, vector figures, camera-ready PDF, and LaTeX source files for the **FairFake** deepfake detection and demographic fairness auditing project.

The formatting strictly mirrors the reference publication:
> **Reference Paper:**  
> *Analyzing Fairness in Deepfake Detection With Massively Annotated Databases*,  
> **IEEE Transactions on Technology and Society**, Vol. 5, No. 1, March 2024.  
> Authors: Ying Xu, Philipp Terhörst, Marius Pedersen, and Kiran Raja.  
> DOI: [10.1109/TTS.2024.3365421](https://doi.org/10.1109/TTS.2024.3365421)

---

## 👥 Student Authors & Researchers
1. **Krishna Yadav** (Roll No: **202451091**), *Student Member, IEEE*
2. **Manush Patel** (Roll No: **202451102**), *Student Member, IEEE*
3. **Dishant** (Roll No: **202451103**), *Student Member, IEEE*

*Department of Computer Science and Engineering, College of Engineering and Technology.*

---

## 📄 Deliverables & Publication Files Overview

1. **`FairFake_IEEE_Paper.pdf`** *(Camera-Ready Publication PDF)*:
   - Full, publication-quality **IEEE Transactions two-column journal paper**.
   - Includes standard IEEE journal masthead (`IEEE TRANSACTIONS ON TECHNOLOGY AND SOCIETY`), alternating running headers with student names, footnotes with student roll numbers, GitHub repository links, DOI, and Creative Commons CC-BY-NC-ND license blocks.
   - Formatted Equations (1)–(5), Algorithm 1 box, Tables I, II, III, and IV, high-resolution vector figures (Figures 1–6 including experimental test cases), structured Key Findings bullets, and Author Biographies with student portrait avatars.

2. **`paper.html`** *(Standalone Interactive & Printable Paper)*:
   - Pixel-perfect IEEE Transactions styling built with standard CSS column layout, justified text, and Times New Roman typography.
   - Includes full-width spanning tables (`TABLE II`, `TABLE IV`), single-column tables (`TABLE I`, `TABLE III`), boxed algorithmic framework, vector charts, experimental test cases (Figure 6), student biographies, and print stylesheet.
   - Includes top navigation bar with one-click print-to-PDF button and links to LaTeX sources and GitHub repository.

3. **`main.tex`** *(Verified IEEE Transactions LaTeX Source File)*:
   - Formatted strictly in **IEEE Transactions journal style** (`\documentclass[journal]{IEEEtran}`).
   - Configured with `\markboth{IEEE TRANSACTIONS ON TECHNOLOGY AND SOCIETY,...}{Yadav, Patel, and Dishant: ...}`.
   - Lists all 3 students with roll numbers and author biographies (`\begin{IEEEbiography}[{\includegraphics[...]{figures/author_*.pdf}}]{...}`).
   - Includes all figures (`figures/fig1_distribution.pdf` through `figures/fig6_test_cases.pdf`).

4. **`references.bib`** *(Complete BibTeX Database)*:
   - Complete BibTeX entries with peer-reviewed citations matching the reference paper and project baselines.

5. **`figures/`** *(Publication-Grade Vector & Bitmap Diagrams)*:
   - `author_krishna.png` & `author_krishna.pdf`: Student portrait avatar for Krishna Yadav (202451091).
   - `author_manush.png` & `author_manush.pdf`: Student portrait avatar for Manush Patel (202451102).
   - `author_dishant.png` & `author_dishant.pdf`: Student portrait avatar for Dishant (202451103).
   - `fig1_distribution.png` & `fig1_distribution.pdf`: Stacked percentage distribution of annotations across attributes for Celeb-DF, FF++, and DFDC.
   - `fig2_correlations.png` & `fig2_correlations.pdf`: Top pairwise Pearson correlations (co-occurring vs. mutually exclusive facial traits).
   - `fig3_rp_crp.png` & `fig3_rp_crp.pdf`: Relative Performance ($RP$) vs. Corrected Relative Performance ($CRP$) with parity bisectrix.
   - `fig4_pdrp_ddrp.png` & `fig4_pdrp_ddrp.pdf`: Four-quadrant disparity decomposition ($PDRP$ vs. $DDRP$) demonstrating false alarms vs. attack evasion vulnerabilities.
   - `fig5_pipeline.png` & `fig5_pipeline.pdf`: FairFake multimodal system architecture with Grad-CAM and DeepFace biometric parsing.
   - `fig6_test_cases.png` & `fig6_test_cases.pdf`: Four experimental test scenarios evaluating real vs. fake predictions, Grad-CAM attention heatmaps, and failure mode analysis.

---

## 🚀 How to Recompile LaTeX (Overleaf or Local)

### Option 1: Overleaf (Zero Setup)
1. Go to [Overleaf](https://www.overleaf.com).
2. Create a **New Project** $\rightarrow$ **Upload Project**.
3. Upload `main.tex`, `references.bib`, and the `figures/` directory.
4. Set compiler to **pdfLaTeX** and click **Recompile**.

### Option 2: Local Compilation
```bash
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
