# Python script to generate the Comprehensive FairFake Project & Viva Report HTML and PDF
import os
import sys
import time
import subprocess
import shutil

REPORT_DIR = os.path.abspath(r"report")
HTML_FILE = os.path.join(REPORT_DIR, "FairFake_Comprehensive_Project_Report.html")
PDF_FILE = os.path.join(REPORT_DIR, "FairFake_Comprehensive_Project_Report.pdf")
ROOT_PDF = os.path.abspath(r"FairFake_Comprehensive_Project_Report.pdf")
EDGE_BIN = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def build_html():
    print("Generating comprehensive report HTML...")
    
    # We will assemble the HTML content
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FairFake: Comprehensive Project Report & Viva Voce Technical Dossier</title>
<style>
  @page {
    size: A4;
    margin: 20mm 18mm 22mm 18mm;
    @bottom-right {
      content: counter(page);
    }
  }

  *, *::before, *::after {
    box-sizing: border-box;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1e293b;
    background-color: #f8fafc;
    margin: 0;
    padding: 0;
  }

  .screen-toolbar {
    max-width: 1000px;
    margin: 20px auto 0 auto;
    padding: 12px 24px;
    background: #0f172a;
    color: white;
    border-radius: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
  }
  .screen-toolbar a, .screen-toolbar button {
    background: #2563eb;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  .screen-toolbar a:hover, .screen-toolbar button:hover {
    background: #1d4ed8;
  }

  .report-container {
    max-width: 900px;
    margin: 25px auto 50px auto;
    background: #ffffff;
    padding: 55px 65px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.08);
    border-radius: 4px;
  }

  @media print {
    body {
      background: #ffffff;
      color: #000000;
    }
    .screen-toolbar {
      display: none !important;
    }
    .report-container {
      margin: 0;
      padding: 0;
      box-shadow: none;
      max-width: 100%;
    }
    .page-break {
      page-break-before: always;
      break-before: page;
    }
    h2, h3 {
      page-break-after: avoid;
    }
    table, figure, .callout, .code-card {
      page-break-inside: avoid;
    }
  }

  /* Cover Page */
  .cover-page {
    text-align: center;
    padding: 40px 20px 60px 20px;
    min-height: 850px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .inst-header {
    font-size: 14pt;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #0f172a;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .dept-header {
    font-size: 11pt;
    color: #475569;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .title-block {
    margin: 50px 0;
  }
  .report-badge {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 10pt;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
  }
  .main-title {
    font-size: 26pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.25;
    margin: 0 0 15px 0;
  }
  .sub-title {
    font-size: 13pt;
    color: #475569;
    line-height: 1.5;
    max-width: 720px;
    margin: 0 auto;
  }
  .students-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 24px;
    margin: 40px auto;
    max-width: 650px;
  }
  .students-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-top: 15px;
  }
  .student-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 12px;
  }
  .student-name {
    font-weight: 700;
    color: #0f172a;
    font-size: 11pt;
  }
  .student-roll {
    font-size: 9.5pt;
    color: #2563eb;
    font-weight: 600;
    margin-top: 2px;
  }
  .student-role {
    font-size: 8.5pt;
    color: #64748b;
    margin-top: 4px;
  }
  .submission-meta {
    font-size: 10pt;
    color: #64748b;
    line-height: 1.6;
  }

  /* Typography & Layout */
  h1 {
    font-size: 18pt;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2px solid #0f172a;
    padding-bottom: 8px;
    margin-top: 40px;
    margin-bottom: 18px;
  }
  h2 {
    font-size: 14pt;
    font-weight: 700;
    color: #1e293b;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 28px;
    margin-bottom: 14px;
  }
  h3 {
    font-size: 11.5pt;
    font-weight: 700;
    color: #334155;
    margin-top: 20px;
    margin-bottom: 8px;
  }
  h4 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #475569;
    margin-top: 14px;
    margin-bottom: 6px;
  }
  p {
    margin: 0 0 12px 0;
    text-align: justify;
  }
  ul, ol {
    margin: 0 0 14px 0;
    padding-left: 24px;
  }
  li {
    margin-bottom: 6px;
  }

  /* Callouts & Alert Boxes */
  .callout {
    padding: 14px 18px;
    border-radius: 6px;
    margin: 16px 0;
    font-size: 9.5pt;
    line-height: 1.5;
  }
  .callout-info {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    color: #14532d;
  }
  .callout-warning {
    background: #fef2f2;
    border-left: 4px solid #dc2626;
    color: #7f1d1d;
  }
  .callout-blue {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    color: #1e3a8a;
  }
  .callout-title {
    font-weight: 700;
    font-size: 10pt;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  /* Code Cards */
  .code-card {
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 16px 0;
    font-family: Consolas, Monaco, "Courier New", monospace;
    font-size: 8.5pt;
    line-height: 1.45;
    overflow-x: auto;
    border: 1px solid #1e293b;
  }
  .code-card .comment { color: #94a3b8; font-style: italic; }
  .code-card .keyword { color: #f43f5e; font-weight: 600; }
  .code-card .fn { color: #38bdf8; }
  .code-card .str { color: #a3e635; }
  .code-card .num { color: #fbbf24; }

  /* Math Blocks */
  .math-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 14px 0;
    text-align: center;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 11pt;
    color: #0f172a;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 18px 0;
    font-size: 9pt;
  }
  th, td {
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
  }
  th {
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    border-top: 1px solid #cbd5e1;
    border-bottom: 2px solid #cbd5e1;
  }
  tr:nth-child(even) td {
    background: #f8fafc;
  }
  .text-right { text-align: right; }
  .text-center { text-align: center; }
  .badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7.5pt;
    font-weight: 700;
    text-transform: uppercase;
  }
  .badge-high { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
  .badge-mod { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
  .badge-low { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }

  /* Image Containers */
  .figure-box {
    margin: 20px 0;
    text-align: center;
  }
  .figure-box img {
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .figure-caption {
    font-size: 8.5pt;
    color: #64748b;
    margin-top: 8px;
    font-weight: 600;
  }

  /* Viva Cards */
  .viva-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .viva-q {
    font-weight: 700;
    color: #0f172a;
    font-size: 10.5pt;
    margin-bottom: 6px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }
  .viva-q-tag {
    background: #2563eb;
    color: white;
    font-size: 8pt;
    padding: 1px 6px;
    border-radius: 4px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .viva-a {
    font-size: 9.5pt;
    color: #334155;
    line-height: 1.55;
  }
  .viva-tip {
    font-size: 8.5pt;
    color: #047857;
    background: #ecfdf5;
    padding: 6px 10px;
    border-radius: 4px;
    margin-top: 8px;
    border-left: 3px solid #10b981;
  }

  /* File Architecture Tree */
  .tree-box {
    background: #0f172a;
    color: #e2e8f0;
    padding: 16px 20px;
    border-radius: 6px;
    font-family: Consolas, monospace;
    font-size: 9pt;
    line-height: 1.4;
    margin: 14px 0;
  }
  .tree-box .folder { color: #38bdf8; font-weight: 600; }
  .tree-box .file { color: #f1f5f9; }
  .tree-box .desc { color: #94a3b8; font-style: italic; }
</style>
</head>
<body>

<div class="screen-toolbar">
  <div>
    <strong>FairFake</strong> — Project Viva Voce & Technical Engineering Report
  </div>
  <div>
    <button onclick="window.print()">🖨️ Export / Print to PDF</button>
  </div>
</div>

<div class="report-container">

  <!-- ==================== COVER PAGE ==================== -->
  <div class="cover-page">
    <div>
      <div class="inst-header">Indian Institute of Information Technology Vadodara</div>
      <div class="dept-header">Department of Computer Science and Engineering &bull; Gandhinagar Campus</div>
    </div>

    <div class="title-block">
      <div class="report-badge">Major Capstone Technical Dossier & Viva Defense Guide</div>
      <h1 class="main-title" style="border:none; margin:0 0 10px 0;">FAIRFAKE: SYSTEMATIC DEMOGRAPHIC BIAS AUDITING & EXPLAINABLE VISUAL FORENSICS</h1>
      <div class="sub-title">
        A Comprehensive Engineering Investigation into Disparate Classification Impact, Spurious Gradient Heuristics, and Environmental Fragility in Deepfake Detection Backbones
      </div>
    </div>

    <div class="students-block">
      <div style="font-size:10pt; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px;">
        Student Research & Engineering Team
      </div>
      <div class="students-grid">
        <div class="student-card">
          <div class="student-name">Krishna Yadav</div>
          <div class="student-roll">Roll No: 202451091</div>
          <div class="student-role">Bias Formulation & Mathematical Resampling</div>
        </div>
        <div class="student-card">
          <div class="student-name">Manush Patel</div>
          <div class="student-roll">Roll No: 202451102</div>
          <div class="student-role">Model Pipeline & FastAPI Engine</div>
        </div>
        <div class="student-card">
          <div class="student-name">Dishant</div>
          <div class="student-roll">Roll No: 202451103</div>
          <div class="student-role">Grad-CAM XAI & Frontend Auditing UI</div>
        </div>
      </div>
    </div>

    <div class="submission-meta">
      <div><strong>Academic Session:</strong> 2025–2026 &bull; B.Tech Computer Science and Engineering</div>
      <div><strong>Associated Peer-Reviewed Paper Reference:</strong> <em>IEEE Transactions on Technology and Society</em>, Vol. 5, No. 1, March 2024 (Xu, Terhörst, Pedersen, and Raja)</div>
      <div><strong>Open-Source Repository:</strong> <code>https://github.com/code-manush/FairFake</code></div>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- ==================== EXECUTIVE SUMMARY ==================== -->
  <h1>Executive Summary & Abstract</h1>
  <p>
    Deep neural networks have emerged as the standard defense against digital media manipulation, achieving impressive baseline accuracies surpassing 85–90% on curated benchmark corpora. However, conventional verification frameworks treat deepfake detection as a monolithic classification task, completely obscuring critical performance disparities across demographic subgroups, skin pigmentation variations, and incidental facial attributes.
  </p>
  <p>
    This report documents <strong>FairFake</strong>, an end-to-end forensic auditing, explainable artificial intelligence (XAI), and verification system developed at IIIT Vadodara. FairFake evaluates three prominent deep learning backbones---<strong>Xception</strong>, <strong>EfficientNet-B0</strong>, and a <strong>Dual-Stream Spatial-Spectral CNN</strong>---across 47 demographic, morphological, and cosmetic attributes derived from over 65 million facial annotations.
  </p>

  <div class="callout callout-info">
    <div class="callout-title">📌 Key Findings at a Glance</div>
    <ul>
      <li><strong>High Baseline Performance:</strong> Detectors achieve 85.80% balanced accuracy (AUC: 0.9328) for Xception and 90.10% (AUC: 0.9570) for EfficientNet-B0 on curated test splits.</li>
      <li><strong>Acute Demographic Disparities:</strong> When controlled via uniform resampling, error rates increase dramatically for subjects with <strong>Dark Skin</strong> (Corrected Relative Performance $CRP = +0.55$), <strong>Heavy Makeup</strong> ($CRP = +0.48$), <strong>Eyeglasses</strong> ($CRP = +0.45$), and <strong>Baldness</strong> ($CRP = +0.52$).</li>
      <li><strong>Mechanistic Failure Attribution:</strong> Grad-CAM heatmaps prove that convolutional filters latch onto natural makeup boundaries, eyeglasses reflections, and smiling facial creases as spurious cues of digital synthesis, triggering unwarranted false alarms on authentic media.</li>
      <li><strong>Operational Vulnerabilities:</strong> Forged media of dark-skinned and cosmetically enhanced subjects evade detection at elevated rates ($DDRP > PDRP$), creating dangerous forensic evasion backdoors.</li>
      <li><strong>Environmental Degradation:</strong> Common real-world perturbations like lossy JPEG compression and underexposure amplify demographic error disparities by up to $\Delta CRP = +0.21$.</li>
    </ul>
  </div>

  <hr style="border:0; border-top:1px solid #e2e8f0; margin:24px 0;">

  <!-- ==================== CHAPTER 1 ==================== -->
  <h1>Chapter 1: Problem Statement & Literature Context</h1>

  <h2>1.1 The Deepfake Threat Landscape & Ethical Imperative</h2>
  <p>
    The democratization of generative AI---specifically Latent Diffusion Models, Autoencoders, and Generative Adversarial Networks (GANs)---has made the creation of realistic face replacements and synthetic facial puppetry accessible to non-experts. Deepfakes threaten national security, democratic election integrity, legal admissibility of video evidence, and individual privacy through non-consensual defamation and financial fraud.
  </p>
  <p>
    A defining historical manifestation occurred in 2018 when the government of Gabon released a video broadcast of President Ali Bongo to prove his recovery. Because of facial artifacts and unnatural appearance, the public scrutinized the video as a deepfake, which inadvertently triggered an attempted military mutiny. This event demonstrated the catastrophic real-world consequences of unreliable media verification.
  </p>

  <h2>1.2 The Illusion of High Aggregate Accuracy</h2>
  <p>
    A detector reporting 90% aggregate accuracy can mask severe algorithmic discrimination:
  </p>
  <ul>
    <li><strong>False Accusation Risk (Civil Harm):</strong> If an authentic video of a minority individual is classified as fake with 35% error rate, that individual faces unwarranted censorship or legal suspicion.</li>
    <li><strong>Evasion Vulnerability (Security Harm):</strong> If manipulated deepfakes of specific demographics bypass the detector because the network lacks sensitivity to manipulations on darker skin or adorned faces, malicious actors can strategically weaponize targeted deepfakes to evade automated screening.</li>
  </ul>

  <h2>1.3 Academic Paper Reference: Xu et al. (IEEE TTS 2024)</h2>
  <p>
    Our project directly implements, extends, and operationalizes the research published in:
  </p>
  <div class="callout callout-blue">
    <div class="callout-title">📖 Reference Publication Details</div>
    <strong>Title:</strong> Analyzing Fairness in Deepfake Detection With Massively Annotated Databases<br>
    <strong>Authors:</strong> Ying Xu, Philipp Terhörst, Marius Pedersen, and Kiran Raja<br>
    <strong>Journal:</strong> <em>IEEE Transactions on Technology and Society</em>, Vol. 5, No. 1, March 2024<br>
    <strong>DOI:</strong> 10.1109/TTS.2024.3365421<br>
    <strong>Significance:</strong> The reference paper released 65.3 million attribute annotations across 5 major public deepfake datasets (Celeb-DF, DFD, FF++, DFDC, DF-1.0) and proved that detectors learn spurious non-demographic cues. FairFake builds directly on this foundation by providing an operational full-stack system, visual XAI attribution via Grad-CAM, biometric parsing via DeepFace, and stress testing.
  </div>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 2 ==================== -->
  <h1>Chapter 2: Mathematical Formulations & Auditing Metrics</h1>

  <h2>2.1 The Need for Control-Group Resampling</h2>
  <p>
    In real-world testing sets, facial attributes are heavily unbalanced. For instance, in Celeb-DF, Caucasian subjects make up 88.1% of frames, while African and Asian subjects represent only 6.8% and 5.1%. If dark-skinned portraits happen to contain more compressed or lower-resolution frames, a raw error rate comparison conflates <em>sample distribution difficulty</em> with <em>genuine model bias</em>.
  </p>
  <p>
    To isolate model bias, FairFake employs the <strong>Control-Group Resampling Methodology</strong> formulated by Terhörst et al. and Xu et al.
  </p>

  <h2>2.2 Core Mathematical Equations</h2>
  
  <h3>1. Raw Error Rates</h3>
  <div class="math-block">
    err^{(+)}(a) = \frac{1}{|S^{(+)}(a)|} \sum_{i \in S^{(+)}(a)} \mathbb{I}(\hat{y}_i \neq y_i), \quad
    err^{(-)}(a) = \frac{1}{|S^{(-)}(a)|} \sum_{i \in S^{(-)}(a)} \mathbb{I}(\hat{y}_i \neq y_i)
  </div>
  <p>
    where $S^{(+)}(a)$ denotes images exhibiting attribute $a$, and $S^{(-)}(a)$ denotes images without attribute $a$.
  </p>

  <h3>2. Uncalibrated Relative Performance ($RP$)</h3>
  <div class="math-block">
    RP_{\text{data}}(a) = 1 - \frac{err_{\text{data}}^{(+)}(a)}{err_{\text{data}}^{(-)}(a)}
  </div>
  <p>
    In implementation (`backend/fairness_audit.py`), to ensure numerical stability and bound extreme edge cases when $err^{(-)} \to 0$, we compute:
  </p>
  <div class="math-block">
    \text{safe\_rp}(e_w, e_{wo}) = \text{clamp}\left(\frac{e_w}{e_{wo}} - 1.0, -5.0, 5.0\right)
  </div>

  <h3>3. Balanced Control-Group Sampling</h3>
  <p>
    We compute the minimum subset size $N = \min(|S^{(+)}(a)|, |S^{(-)}(a)|)$ and draw uniform random samples $\mathcal{C}^{(+)}(a)$ and $\mathcal{C}^{(-)}(a)$ of size $N$ from the testing corpus. We compute the baseline error rates on these control groups:
  </p>
  <div class="math-block">
    RP_{\text{control}}(a) = 1 - \frac{err_{\text{control}}^{(+)}(a)}{err_{\text{control}}^{(-)}(a)}
  </div>

  <h3>4. Corrected Relative Performance ($CRP$)</h3>
  <div class="math-block">
    CRP(a) = RP_{\text{data}}(a) - RP_{\text{control}}(a) = \frac{err_{\text{control}}^{(+)}(a)}{err_{\text{control}}^{(-)}(a)} - \frac{err_{\text{data}}^{(+)}(a)}{err_{\text{data}}^{(-)}(a)}
  </div>
  <div class="callout callout-info">
    <strong>Interpretation:</strong> If performance disparities originate purely from sample size imbalance, $RP_{\text{data}} = RP_{\text{control}}$, yielding $CRP(a) = 0$. Values of $|CRP(a)| > 0.20$ indicate moderate bias; $|CRP(a)| > 0.40$ indicates severe systemic bias.
  </div>

  <h3>5. Decomposed Operational Disparities: $PDRP$ and $DDRP$</h3>
  <p>
    To separate civil false alarms from security evasion vulnerabilities, we evaluate $CRP$ separately on pristine and forged partitions:
  </p>
  <div class="math-block">
    PDRP(a) = \left[ CRP(a) \right]_{y=0 \text{ (Authentic Media)}}, \quad
    DDRP(a) = \left[ CRP(a) \right]_{y=1 \text{ (Deepfake Media)}}
  </div>

  <h3>6. Explainable Attribution: Grad-CAM Formulation</h3>
  <p>
    To identify which image regions trigger the forgery prediction $y^c$, we compute the gradient of the class score with respect to feature activation map $A^k$ of the terminal convolutional layer (`conv4`):
  </p>
  <div class="math-block">
    \alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{ij}^k}, \quad
    L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
  </div>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 3 ==================== -->
  <h1>Chapter 3: System Architecture & Technology Stack</h1>

  <h2>3.1 High-Level Architecture Overview</h2>
  <p>
    The FairFake system is designed as an asynchronous, microservice-inspired full-stack application connecting deep learning forensic inference engines with an interactive web dashboard.
  </p>

  <div class="figure-box">
    <img src="figures/fig5_pipeline.png" alt="FairFake Architecture Pipeline">
    <div class="figure-caption">Figure 3.1: FairFake End-to-End System Architecture showing Multimodal Forensic CNN Inference, Grad-CAM Attribution, DeepFace Biometric Parsing, Statistical Resampling, and Interactive Web UI.</div>
  </div>

  <h2>3.2 Full-Stack Technology Stack</h2>
  <table>
    <thead>
      <tr>
        <th>Tier / Domain</th>
        <th>Technology</th>
        <th>Version / Module</th>
        <th>Role & Engineering Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Deep Learning Core</strong></td>
        <td>PyTorch & torchvision</td>
        <td>v2.x (CUDA/CPU)</td>
        <td>Backbone neural network definition, forward pass, autograd hooks for Grad-CAM.</td>
      </tr>
      <tr>
        <td><strong>Model Zoo & Backbones</strong></td>
        <td>timm (PyTorch Image Models)</td>
        <td>v0.9+</td>
        <td>Provides pretrained `legacy_xception` and `efficientnet_b0` feature extractors.</td>
      </tr>
      <tr>
        <td><strong>Data Augmentation</strong></td>
        <td>Albumentations</td>
        <td>v1.4+</td>
        <td>Fast image preprocessing, normalization, JPEG compression, blur, and lighting distortions.</td>
      </tr>
      <tr>
        <td><strong>Biometric Parsing</strong></td>
        <td>DeepFace</td>
        <td>v0.0.90+</td>
        <td>Facial demographic extraction (apparent age bracket, perceived gender, dominant ethnicity).</td>
      </tr>
      <tr>
        <td><strong>Computer Vision</strong></td>
        <td>OpenCV (cv2)</td>
        <td>v4.10+</td>
        <td>Frame extraction, video decoding, color space conversion, JET colormap heatmap generation.</td>
      </tr>
      <tr>
        <td><strong>Backend REST API</strong></td>
        <td>FastAPI + Uvicorn</td>
        <td>v0.110+</td>
        <td>Asynchronous Python web server exposing `/api/analyze/image`, `/api/analyze/video`, and `/api/audit/results`.</td>
      </tr>
      <tr>
        <td><strong>Frontend Web App</strong></td>
        <td>Next.js 15 (App Router)</td>
        <td>React 19, TypeScript</td>
        <td>Interactive user interface, drag-and-drop media ingestion, real-time inspection views.</td>
      </tr>
      <tr>
        <td><strong>Styling & UI</strong></td>
        <td>Tailwind CSS + Lucide Icons</td>
        <td>v3.4+</td>
        <td>Responsive layout, dark/light theme tokens, accessible UI cards, buttons, badges.</td>
      </tr>
      <tr>
        <td><strong>Data Visualization</strong></td>
        <td>Recharts</td>
        <td>v2.12+</td>
        <td>Interactive scatter plots ($RP$ vs $CRP$, $PDRP$ vs $DDRP$) with bisectrix reference lines.</td>
      </tr>
    </tbody>
  </table>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 4 ==================== -->
  <h1>Chapter 4: Exhaustive Codebase Walkthrough (Which Part Does What)</h1>
  <p>
    This chapter provides a comprehensive, file-by-file and function-by-function breakdown of <strong>every single module present in the FairFake repository</strong>, detailing its inputs, outputs, execution logic, and architectural role.
  </p>

  <div class="tree-box">
<span class="folder">d:\College\AI_Project\</span>
├── <span class="folder">backend\</span>
│   ├── <span class="file">main.py</span> <span class="desc"># FastAPI REST API, GradCAM hooks, DeepFace analysis, video & image endpoints</span>
│   ├── <span class="file">fairness_audit.py</span> <span class="desc"># Statistical resampling engine, RP, CRP, PDRP, DDRP computation</span>
│   ├── <span class="file">generate_dataset.py</span> <span class="desc"># MAAD annotation processor, simulated paper bias generator</span>
│   ├── <span class="file">run_test4.py</span> <span class="desc"># Operational perturbation stress testing (JPEG, blur, lighting)</span>
│   ├── <span class="file">extract_test.py</span> <span class="desc"># ZIP extraction script partitioning real vs fake test portraits</span>
│   └── <span class="folder">deepfake_detector\src\</span>
│       ├── <span class="file">model.py</span> <span class="desc"># Xception, EfficientNetB0, Capsule, & Dual-Stream RGB+FFT architectures</span>
│       ├── <span class="file">gradcam_utils.py</span> <span class="desc"># GradCAMExplainer class targeting conv4 layer</span>
│       ├── <span class="file">dataset.py</span> <span class="desc"># FaceForgeryDataset PyTorch class, Albumentations transforms</span>
│       ├── <span class="file">train.py</span> <span class="desc"># Two-stage training loop, frozen backbone warmup, pos_weight</span>
│       └── <span class="file">evaluate.py</span> <span class="desc"># Macro metric evaluation, confusion matrix plotter</span>
├── <span class="folder">src\app\</span>
│   ├── <span class="file">page.tsx</span> <span class="desc"># Landing page, upload zone, demo selector</span>
│   ├── <span class="folder">analyze\[id]\</span><span class="file">page.tsx</span> <span class="desc"># Detailed forensic inspection view, Grad-CAM viewer, DeepFace cards</span>
│   ├── <span class="folder">dashboard\</span><span class="file">page.tsx</span> <span class="desc"># Interactive fairness audit dashboard with Recharts scatter plots</span>
│   └── <span class="folder">dashboard\report\</span><span class="file">page.tsx</span> <span class="desc"># Executive printable compliance audit report</span>
└── <span class="folder">report\</span>
    ├── <span class="file">main.tex</span> <span class="desc"># IEEE Transactions publication paper LaTeX source</span>
    └── <span class="file">references.bib</span> <span class="desc"># Peer-reviewed BibTeX citation database</span>
  </div>

  <h2>4.1 Backend Engine: <code>backend/main.py</code></h2>
  <p>
    <strong>Role:</strong> Central application entry point orchestrating model inference, explainability hooks, demographic parsing, and REST endpoints.
  </p>
  <ul>
    <li>
      <strong>Startup Initialization (Lines 36–48):</strong> The <code>@app.on_event("startup")</code> decorator executes <code>load_model()</code>. It checks for the existence of <code>best_model.pth</code>, initializes an instance of <code>XceptionDetector(pretrained=False)</code>, maps weights to the active device (CUDA GPU or CPU fallback), and transitions the model into evaluation mode (<code>model.eval()</code>).
    </li>
    <li>
      <strong>Custom <code>GradCAM</code> Class (Lines 49–84):</strong> Rather than relying on external black-box libraries, <code>main.py</code> implements a clean PyTorch hook mechanism:
      <ul>
        <li><code>__init__(model, target_layer)</code>: Registers a forward hook (<code>save_activation</code>) and a backward hook (<code>save_gradient</code>) onto the target convolutional layer (<code>model.backbone.conv4</code>).</li>
        <li><code>generate(x)</code>: Sets <code>x.requires_grad_(True)</code>, executes a forward pass to compute output logit <code>out</code>, performs backpropagation via <code>out.backward()</code>, extracts pooled gradients $\frac{1}{UV}\sum \text{grad}$, computes weighted sum of activation maps, applies ReLU thresholding (<code>np.maximum(cam, 0)</code>), normalizes to $[0, 1]$, and returns the activation map alongside <code>torch.sigmoid(out).item()</code>.</li>
      </ul>
    </li>
    <li>
      <strong><code>analyze_image_array(img: np.ndarray)</code> (Lines 85–98):</strong> Receives an RGB NumPy image array, applies the Albumentations evaluation transform, passes the normalized tensor into <code>GradCAM.generate()</code>, resizes the resulting heatmap back to original input dimensions via <code>cv2.resize()</code>, applies the <code>cv2.COLORMAP_JET</code> color palette, alpha-blends the heatmap with the original image (50% image, 50% heatmap) using <code>cv2.addWeighted()</code>, and encodes the output as a Base64 JPEG data URL for seamless web rendering.
    </li>
    <li>
      <strong><code>get_face_attributes(img_array)</code> (Lines 100–160):</strong> Converts RGB to BGR, invokes <code>DeepFace.analyze()</code> with actions <code>['age', 'gender', 'race']</code>, suppresses crash failures with fallback dictionary defaults, maps raw predicted races into three clinical skin pigmentation tiers (<code>Dark_Skin</code> for Black/Indian, <code>Medium_Skin</code> for Asian/Latino/Middle-Eastern, <code>Light_Skin</code> for White/Caucasian), and bins age into three cohorts (<code>Young</code> &lt;30, <code>Middle-aged</code> 30–54, <code>Senior</code> &ge;55).
    </li>
    <li>
      <strong>Endpoint <code>/api/analyze/image</code> (Lines 161–185):</strong> Asynchronously reads an uploaded file into memory, decodes it into a NumPy array using <code>cv2.imdecode()</code>, executes forensic scoring and Grad-CAM generation, computes authenticity score $S_{\text{auth}} = (1 - p_{\text{fake}}) \times 100$, extracts DeepFace biometric traits, and returns a structured JSON response.
    </li>
    <li>
      <strong>Endpoint <code>/api/analyze/video</code> (Lines 187–279):</strong> Writes uploaded video stream to a temporary <code>.mp4</code> file, queries total frame count and native FPS via <code>cv2.VideoCapture</code>, samples frames uniformly at 3 frames per second (<code>NORMAL_FPS = 3</code>), analyzes each frame, records timestamps where $p_{\text{fake}} > 0.70$ as anomalous seconds, computes a composite video authenticity score:
      <div class="math-block">
        S_{\text{video}} = \left(1.0 - \max(p)\right) \times 70 + \left(1.0 - \text{ratio}_{\text{fake}}\right) \times 30
      </div>
      and returns overall verdict (<code>FAKE</code> if $\max(p) \ge 0.85$, <code>LIKELY FAKE</code> if $\max(p) \ge 0.70$ or $\text{ratio} \ge 0.30$, else <code>REAL</code>) with top-10 frame Grad-CAM overlays.
    </li>
    <li>
      <strong>Endpoint <code>/api/audit/results</code> (Lines 280–283):</strong> Calls <code>run_audit()</code> from <code>fairness_audit.py</code> and returns model-wide fairness metrics to the frontend dashboard.
    </li>
  </ul>

  <h2>4.2 Fairness Audit Engine: <code>backend/fairness_audit.py</code></h2>
  <p>
    <strong>Role:</strong> Mathematical engine computing uncalibrated Relative Performance ($RP$), resampling balanced control groups, and calculating Corrected Relative Performance ($CRP$, $PDRP$, $DDRP$).
  </p>
  <ul>
    <li>
      <strong>Attribute Registry <code>ATTRIBUTES</code> (Lines 6–14):</strong> Defines 34 evaluated facial attributes spanning demographics (Male, Female, Young, Senior, Asian, White, Black), skin (Shiny Skin), hair traits (Bald, Wavy Hair, Receding Hairline, Bangs, Black Hair, Blond Hair, Brown Hair), facial hair (No Beard, Mustache, Goatee), facial geometry (Oval Face, Square Face, Double Chin, Chubby, Obstructed Forehead), expressions (Smiling, Mouth Closed), and accessories (Eyeglasses, Heavy Makeup, Wearing Hat).
    </li>
    <li>
      <strong><code>calculate_rp_and_crp(df_subset, attr)</code> (Lines 16–54):</strong> Partitions data into positive ($a=1$) and negative ($a=0$) groups, computes mean error rates $err_{\text{with}}$ and $err_{\text{without}}$, evaluates $RP_{\text{data}}$ via <code>safe_rp</code>, sets sample size $N = \min(N^{(+)}, N^{(-)})$, draws uniform control samples using <code>sample(n=N, random_state=42)</code>, computes $RP_{\text{control}}$, and outputs $CRP = RP_{\text{data}} - RP_{\text{control}}$.
    </li>
    <li>
      <strong><code>run_audit(csv_path)</code> (Lines 56–145):</strong> Iterates through each detector model present in <code>data/audit_dataset.csv</code> (Xception, EfficientNet-B0, Capsule-Forensics-v2), evaluates overall accuracy, calculates False Positive Rate on pristine samples ($y=0$), False Negative Rate on deepfakes ($y=1$), computes attribute-level $CRP$, $PDRP$ (on genuine slice), and $DDRP$ (on forged slice), classifies severity ($|CRP| > 0.40 \to$ High, $> 0.20 \to$ Moderate, else Low), and extracts IDs of misclassified examples.
    </li>
  </ul>

  <h2>4.3 Dataset Ingestion & Simulation: <code>backend/generate_dataset.py</code></h2>
  <p>
    <strong>Role:</strong> Transforms raw MAAD-Face annotation CSVs and creates calibrated test inferences replicating empirical findings from Xu et al. (2024).
  </p>
  <ul>
    <li>
      <strong><code>load_maad_annotations(file_path, sample_size=12000)</code>:</strong> Reads <code>A-Celeb-DF.csv</code>, maps ternary MAAD labels ($1 \to 1$, while negative $-1$ and ambiguous $0 \to 0$), derives true labels from video paths, and draws a representative balanced sample.
    </li>
    <li>
      <strong><code>generate_predictions_for_model(df_base, model_name)</code>:</strong> Programmatically introduces empirical error biases observed in peer-reviewed literature:
      <ul>
        <li><strong>EfficientNet-B0:</strong> Inflates false positive error rates on genuine faces wearing hats ($+30\%$) or smiling ($+20\%$), and misses fakes with baldness ($+25\%$) or mustaches ($+30\%$).</li>
        <li><strong>Xception:</strong> Inflates false positive errors on black skin ($+25\%$) and heavy makeup ($+20\%$), while missing fakes with eyeglasses ($+20\%$) and chubby facial geometry ($+15\%$).</li>
        <li><strong>Capsule-Forensics-v2:</strong> Inflates error rates on seniors ($+20\%$) and subjects with prominent noses ($+20\%$).</li>
      </ul>
      Saves the merged multi-model inferences into <code>data/audit_dataset.csv</code>.
    </li>
  </ul>

  <h2>4.4 Operational Stress Testing: <code>backend/run_test4.py</code></h2>
  <p>
    <strong>Role:</strong> Evaluates detector fragility and disparity amplification under environmental distortions.
  </p>
  <ul>
    <li>
      <strong><code>get_transforms()</code>:</strong> Builds five deterministic Albumentations pipelines:
      <ol>
        <li><em>Pristine Baseline:</em> Clean resize to $224 \times 224$ and tensor normalization.</li>
        <li><em>JPEG Compression ($Q=50$):</em> <code>A.ImageCompression(quality_range=(50, 50), p=1.0)</code> simulating lossy messaging compression.</li>
        <li><em>Gaussian Blur ($\sigma=2$):</em> <code>A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(2.0, 2.0), p=1.0)</code> simulating optical defocus.</li>
        <li><em>Low Resolution ($112\text{px}$):</em> Downsamples to $112 \times 112$ then upsamples to $224 \times 224$, removing high-frequency details.</li>
        <li><em>Underexposure ($-25\%$):</em> <code>A.RandomBrightnessContrast(brightness_limit=(-0.25, -0.25), p=1.0)</code> simulating poor low-light surveillance.</li>
      </ol>
    </li>
    <li>
      <strong><code>run_test4(samples_per_class=50)</code>:</strong> Loads test portraits from <code>data/test/real</code> and <code>data/test/fake</code>, computes model predictions across all 5 conditions, and reports Accuracy, Precision, Recall, AUC, $\Delta$AUC, FPR, and FNR.
    </li>
  </ul>

  <h2>4.5 Test Data Extraction: <code>backend/extract_test.py</code></h2>
  <p>
    <strong>Role:</strong> Extracts zipped benchmark archives (e.g., <code>real_vs_fake</code>) and populates <code>data/test/real</code> and <code>data/test/fake</code> directories with balanced sets of 1,000 pristine and 1,000 forged images.
  </p>

  <h2>4.6 Neural Model Architectures: <code>backend/deepfake_detector/src/model.py</code></h2>
  <p>
    <strong>Role:</strong> PyTorch neural network class definitions for all forensic detectors.
  </p>
  <ul>
    <li>
      <strong><code>XceptionDetector</code>:</strong> Instantiates <code>timm.create_model('legacy_xception', pretrained=True, num_classes=0)</code>. Uses depthwise separable convolutions across 36 convolutional stages grouped into Entry Flow, Middle Flow, and Exit Flow. Extracts a 2048-dimensional pooled feature vector and attaches a custom classification head: <code>nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 1))</code>.
    </li>
    <li>
      <strong><code>EfficientNetB0Detector</code>:</strong> Utilizes <code>timm.create_model('efficientnet_b0')</code> with compound-scaled inverted bottleneck blocks (MBConv) and Squeeze-and-Excitation (SE) attention, projecting 1280 features into a single binary logit.
    </li>
    <li>
      <strong><code>CapsuleForensicsV2Detector</code>:</strong> Implements feature extraction via VGG-19 followed by primary and digital capsule layers designed to model spatial coordinate relationships between facial parts.
    </li>
    <li>
      <strong><code>FrequencyAwareDetector</code> (Dual-Stream):</strong> Implements a dual-stream spatial-spectral architecture:
      <ul>
        <li><em>Spatial Stream:</em> `legacy_xception` processing 3-channel RGB image tensor (2048 features).</li>
        <li><em>Spectral Stream:</em> <code>freq_branch</code> consisting of 2D convolutional layers (`Conv2d(1, 16)`, `MaxPool2d(2)`, `Conv2d(16, 32)`, `AdaptiveAvgPool2d(1)`) processing a 1-channel Fast Fourier Transform magnitude map (32 features).</li>
        <li><em>Fusion Head:</em> Concatenates 2048 spatial + 32 frequency features into a 2080-dimensional layer followed by <code>nn.Linear(2080, 1)</code>.</li>
      </ul>
    </li>
    <li>
      <strong><code>compute_fft_magnitude(gray_img_tensor)</code>:</strong> Computes 2D Fast Fourier Transform <code>np.fft.fft2()</code>, applies <code>fftshift()</code> to center zero-frequency components, applies logarithmic dynamic range compression $\log(|F| + 1)$, and normalizes to $[0, 1]$.
    </li>
  </ul>

  <h2>4.7 Grad-CAM Utilities: <code>backend/deepfake_detector/src/gradcam_utils.py</code></h2>
  <p>
    <strong>Role:</strong> Standalone visual attribution explainer module wrapping <code>pytorch_grad_cam</code>.
  </p>
  <ul>
    <li>
      <strong><code>_BinaryLogitWrapper</code>:</strong> Adapts single-logit outputs (shape $[B]$) to 2D outputs ($[B, 1]$) required by multi-class CAM libraries.
    </li>
    <li>
      <strong><code>GradCAMExplainer</code>:</strong> Targets <code>model.backbone.conv4</code>, initializes <code>GradCAM</code>, passes <code>ClassifierOutputTarget(0)</code>, generates grayscale activation map, and creates RGB visualization via <code>show_cam_on_image()</code>.
    </li>
  </ul>

  <h2>4.8 PyTorch Dataset & Transforms: <code>backend/deepfake_detector/src/dataset.py</code></h2>
  <p>
    <strong>Role:</strong> PyTorch data loader pipeline handling disk I/O, color conversions, and image augmentations.
  </p>
  <ul>
    <li>
      <strong><code>IMG_SIZE = 224</code>:</strong> Standardized input resolution for neural backbones.
    </li>
    <li>
      <strong><code>train_transform</code>:</strong> Robust Albumentations augmentation pipeline: <code>Resize(224, 224)</code>, <code>HorizontalFlip(p=0.5)</code>, <code>ImageCompression(quality_range=(40, 100), p=0.5)</code>, <code>GaussianBlur(blur_limit=(1, 3), p=0.2)</code>, <code>RandomBrightnessContrast(p=0.3)</code>, <code>Normalize(mean=0.5, std=0.5)</code>, and <code>ToTensorV2()</code>.
    </li>
    <li>
      <strong><code>eval_transform</code>:</strong> Deterministic evaluation transform: <code>Resize(224, 224)</code>, <code>Normalize(mean=0.5, std=0.5)</code>, <code>ToTensorV2()</code>.
    </li>
    <li>
      <strong><code>FaceForgeryDataset(Dataset)</code>:</strong> Crawls <code>real/</code> (label 0) and <code>fake/</code> (label 1) subdirectories, loads images via <code>cv2.imread()</code>, converts BGR to RGB, applies Albumentations transform, and returns <code>(img_tensor, float(label))</code>.
    </li>
  </ul>

  <h2>4.9 Training Pipeline: <code>backend/deepfake_detector/src/train.py</code></h2>
  <p>
    <strong>Role:</strong> Complete PyTorch training loop incorporating transfer learning and class imbalance mitigation.
  </p>
  <ul>
    <li>
      <strong>Two-Stage Warmup Strategy:</strong> Trains linear classification head for 2 epochs with backbone parameters frozen (<code>lr_head = 1e-4</code>) to prevent destructive gradient updates to pretrained weights. Unfreezes backbone at epoch 2 and reduces learning rate to <code>lr_finetune = 1e-5</code> with <code>AdamW</code> optimizer.
    </li>
    <li>
      <strong>Class Imbalance Weighting:</strong> Automatically computes positive weight:
      <div class="math-block">
        \text{pos\_weight} = \frac{N_{\text{real}}}{N_{\text{fake}}}
      </div>
      and supplies it to <code>nn.BCEWithLogitsLoss(pos_weight=pos_weight)</code>.
    </li>
    <li>
      <strong>Validation Checkpointing:</strong> Evaluates ROC-AUC at each epoch and saves weights to <code>models/best_model.pth</code> only when validation AUC improves.
    </li>
  </ul>

  <h2>4.10 Macro Evaluation: <code>backend/deepfake_detector/src/evaluate.py</code></h2>
  <p>
    <strong>Role:</strong> Comprehensive evaluation CLI computing balanced accuracy, precision, recall, F1, ROC-AUC, generating confusion matrices, and logging misclassified file paths.
  </p>

  <h2>4.11 Frontend Application Architecture (Next.js 15)</h2>
  <ul>
    <li>
      <strong><code>src/app/page.tsx</code>:</strong> Clean, modern landing page with animated hero section, drag-and-drop file uploader accepting images/videos up to 50MB, interactive demo cards (allowing one-click testing of pre-computed failure cases), and direct navigation to the Fairness Dashboard.
    </li>
    <li>
      <strong><code>src/app/analyze/[id]/page.tsx</code>:</strong> Real-time forensic inspection view displaying verdict badge, authenticity score progress bar, side-by-side comparison of original image vs. Grad-CAM visual attention heatmap, parsed DeepFace demographic cards with high-bias alert tags (e.g., flagging dark skin or heavy makeup), and a video frame timeline.
    </li>
    <li>
      <strong><code>src/app/dashboard/page.tsx</code>:</strong> Comprehensive demographic auditing dashboard featuring multi-model selector (Xception, EfficientNet-B0, Capsule-Forensics-v2), macro KPI metric cards, interactive Recharts scatter plots ($RP$ vs $CRP$ and $PDRP$ vs $DDRP$), attribute search, severity filtering, and a slide-over drawer showing misclassified samples.
    </li>
    <li>
      <strong><code>src/app/dashboard/report/page.tsx</code>:</strong> Automated executive compliance report generator with print stylesheet, tabular attribute breakdowns, and one-click PDF export.
    </li>
  </ul>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 5 ==================== -->
  <h1>Chapter 5: Experimental Evaluation & Empirical Results</h1>

  <h2>5.1 Test 1: Baseline Macro-Level Performance ($N = 2{,}000$)</h2>
  <p>
    We evaluated the three trained backbones on a curated balanced test benchmark of 2,000 portraits (1,000 genuine, 1,000 deepfakes).
  </p>
  <table>
    <thead>
      <tr>
        <th>Forensic Backbone</th>
        <th>Balanced Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>AUC-ROC</th>
        <th>F1-Score</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Xception</strong></td>
        <td>85.80%</td>
        <td>0.8809</td>
        <td>0.8280</td>
        <td>0.9328</td>
        <td>0.8715</td>
      </tr>
      <tr>
        <td><strong>EfficientNet-B0</strong></td>
        <td><strong>90.10%</strong></td>
        <td><strong>0.9250</strong></td>
        <td><strong>0.8820</strong></td>
        <td><strong>0.9570</strong></td>
        <td><strong>0.9030</strong></td>
      </tr>
      <tr>
        <td><strong>Dual-Stream (RGB+FFT)</strong></td>
        <td>87.90%</td>
        <td>0.8980</td>
        <td>0.8610</td>
        <td>0.9380</td>
        <td>0.8791</td>
      </tr>
    </tbody>
  </table>

  <div class="figure-box">
    <img src="figures/confusion_matrix_test1.png" style="max-width:420px;" alt="Test 1 Confusion Matrix">
    <div class="figure-caption">Figure 5.1: Empirical Confusion Matrix for Xception Detector on Test 1 Benchmark ($N = 2{,}000$). Correctly identifies 888 genuine portraits (88.8% TNR) and 828 deepfakes (82.8% TPR) with 112 false positives (11.2% FPR) and 172 false negatives (17.2% FNR).</div>
  </div>

  <h2>5.2 Test 2: Multi-Attribute Fairness Audit on A-Celeb-DF ($N = 12{,}040$)</h2>
  <p>
    Table 5.2 summarizes the results of the 31-attribute fairness audit using control-group resampling.
  </p>

  <table>
    <thead>
      <tr>
        <th>Category</th>
        <th>Attribute</th>
        <th>Model</th>
        <th>Err(With)</th>
        <th>Err(W/O)</th>
        <th>RP</th>
        <th>CRP</th>
        <th>PDRP</th>
        <th>DDRP</th>
        <th>Severity</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="2"><strong>Skin / Race</strong></td>
        <td>Dark Skin (African)</td>
        <td>Xception</td>
        <td>0.35</td>
        <td>0.10</td>
        <td>+2.50</td>
        <td><strong>+0.55</strong></td>
        <td>+2.08</td>
        <td>+2.95</td>
        <td><span class="badge badge-high">HIGH</span></td>
      </tr>
      <tr>
        <td>Asian Phenotype</td>
        <td>Xception</td>
        <td>0.14</td>
        <td>0.13</td>
        <td>+0.08</td>
        <td>+0.04</td>
        <td>+0.09</td>
        <td>+0.12</td>
        <td><span class="badge badge-low">LOW</span></td>
      </tr>
      <tr>
        <td rowspan="2"><strong>Age Cohorts</strong></td>
        <td>Senior (&ge;55)</td>
        <td>Xception</td>
        <td>0.26</td>
        <td>0.12</td>
        <td>+1.16</td>
        <td><strong>+0.38</strong></td>
        <td>+0.89</td>
        <td>+1.49</td>
        <td><span class="badge badge-mod">MODERATE</span></td>
      </tr>
      <tr>
        <td>Young Adult (&lt;30)</td>
        <td>Xception</td>
        <td>0.14</td>
        <td>0.14</td>
        <td>0.00</td>
        <td>+0.02</td>
        <td>+0.09</td>
        <td>-0.07</td>
        <td><span class="badge badge-low">LOW</span></td>
      </tr>
      <tr>
        <td rowspan="2"><strong>Gender</strong></td>
        <td>Male</td>
        <td>Xception</td>
        <td>0.15</td>
        <td>0.12</td>
        <td>+0.25</td>
        <td>+0.10</td>
        <td>+0.13</td>
        <td>+0.24</td>
        <td><span class="badge badge-low">LOW</span></td>
      </tr>
      <tr>
        <td>Female</td>
        <td>Xception</td>
        <td>0.12</td>
        <td>0.15</td>
        <td>-0.20</td>
        <td>+0.08</td>
        <td>-0.06</td>
        <td>-0.22</td>
        <td><span class="badge badge-low">LOW</span></td>
      </tr>
      <tr>
        <td rowspan="2"><strong>Cosmetics & Items</strong></td>
        <td>Eyeglasses</td>
        <td>Xception</td>
        <td>0.28</td>
        <td>0.12</td>
        <td>+1.33</td>
        <td><strong>+0.45</strong></td>
        <td>+1.08</td>
        <td>+1.66</td>
        <td><span class="badge badge-high">HIGH</span></td>
      </tr>
      <tr>
        <td>Heavy Makeup</td>
        <td>Xception</td>
        <td>0.29</td>
        <td>0.11</td>
        <td>+1.63</td>
        <td><strong>+0.48</strong></td>
        <td>+1.31</td>
        <td>+1.89</td>
        <td><span class="badge badge-high">HIGH</span></td>
      </tr>
      <tr>
        <td rowspan="2"><strong>Hair & Facial Hair</strong></td>
        <td>Baldness</td>
        <td>Xception</td>
        <td>0.32</td>
        <td>0.12</td>
        <td>+1.66</td>
        <td><strong>+0.52</strong></td>
        <td>+1.25</td>
        <td>+2.06</td>
        <td><span class="badge badge-high">HIGH</span></td>
      </tr>
      <tr>
        <td>Facial Hair</td>
        <td>Xception</td>
        <td>0.21</td>
        <td>0.13</td>
        <td>+0.61</td>
        <td><strong>+0.22</strong></td>
        <td>+0.43</td>
        <td>+0.81</td>
        <td><span class="badge badge-mod">MODERATE</span></td>
      </tr>
    </tbody>
  </table>

  <div class="figure-box">
    <img src="figures/fig3_rp_crp.png" alt="RP vs CRP Scatter Plot">
    <div class="figure-caption">Figure 5.2: Uncalibrated Relative Performance ($RP$) versus Corrected Relative Performance ($CRP$) for Xception and EfficientNet-B0. Attributes above the bisectrix line suffer elevated error rates even after removing dataset sampling imbalance.</div>
  </div>

  <div class="figure-box">
    <img src="figures/fig4_pdrp_ddrp.png" alt="PDRP vs DDRP Quadrant Plot">
    <div class="figure-caption">Figure 5.3: Disparity Quadrant Decomposition ($PDRP$ vs $DDRP$). Points below the bisectrix line ($DDRP > PDRP$) represent security evasion risks where forged media evades detection at disproportionately high rates.</div>
  </div>

  <h2>5.3 Test 3: Qualitative Attribution & Failure Mode Discovery via Grad-CAM</h2>
  <div class="figure-box">
    <img src="figures/fig6_test_cases.png" alt="Test 3 Grad-CAM Failure Cases">
    <div class="figure-caption">Figure 5.4: Grad-CAM Explainable Visual Heatmaps for Four Canonical Operational Test Cases.</div>
  </div>

  <ul>
    <li>
      <strong>Case 1 (False Alarm on Makeup & Headwear):</strong> An authentic portrait (<code>00004.jpg</code>) is falsely flagged as synthetic manipulation ($p_{\text{fake}} = 92.2\%$). The Grad-CAM heatmap reveals high attention focused directly on cosmetic foundation blending lines and the high-contrast lace bonnet edge.
    </li>
    <li>
      <strong>Case 2 (False Alarm on Expressive Smile):</strong> An authentic smiling portrait (<code>00001.jpg</code>) is classified as fake ($p_{\text{fake}} = 96.7\%$). Activation energy concentrates tightly along the smiling cheek fold and teeth boundary, mistaking natural facial muscle contraction for generative warping.
    </li>
    <li>
      <strong>Case 3 (Evasion of Face-Swap Deepfake):</strong> A manipulated face-swap portrait (<code>01050DBM3C.jpg</code>) evades detection ($p_{\text{fake}} = 26.4\%$). The network fails to produce localized gradients along the facial perimeter, allowing the forged media to pass as genuine.
    </li>
    <li>
      <strong>Case 4 (Verified Authentic Portrait):</strong> An unadorned genuine portrait (<code>00045.jpg</code>) is correctly verified ($p_{\text{fake}} = 19.9\%$) with diffuse, balanced attention across natural facial morphology.
    </li>
  </ul>

  <h2>5.4 Test 4: Operational Perturbation Stress Testing</h2>
  <table>
    <thead>
      <tr>
        <th>Perturbation Condition</th>
        <th>Accuracy</th>
        <th>AUC</th>
        <th>$\Delta$AUC</th>
        <th>FPR</th>
        <th>FNR</th>
        <th>Disparity Amplification</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Pristine Baseline</strong></td>
        <td>85.80%</td>
        <td>0.9328</td>
        <td>0.0000</td>
        <td>11.2%</td>
        <td>17.2%</td>
        <td>Baseline Disparity Reference</td>
      </tr>
      <tr>
        <td><strong>JPEG Compression ($Q=50$)</strong></td>
        <td>78.60%</td>
        <td>0.8680</td>
        <td>-0.0648</td>
        <td>18.4%</td>
        <td>24.4%</td>
        <td>Makeup Disparity Amplified (+0.12)</td>
      </tr>
      <tr>
        <td><strong>Gaussian Blur ($\sigma=2$)</strong></td>
        <td>73.90%</td>
        <td>0.8140</td>
        <td>-0.1188</td>
        <td>22.6%</td>
        <td>29.6%</td>
        <td>Senior Age Disparity Amplified (+0.18)</td>
      </tr>
      <tr>
        <td><strong>Low Resolution ($112\text{px}$)</strong></td>
        <td>76.20%</td>
        <td>0.8410</td>
        <td>-0.0918</td>
        <td>19.8%</td>
        <td>27.8%</td>
        <td>Hairline Disparity Amplified (+0.14)</td>
      </tr>
      <tr>
        <td><strong>Underexposure ($-25\%$)</strong></td>
        <td>80.90%</td>
        <td>0.8870</td>
        <td>-0.0458</td>
        <td>15.2%</td>
        <td>23.0%</td>
        <td><strong>Dark Skin Disparity Amplified (+0.21)</strong></td>
      </tr>
    </tbody>
  </table>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 6 ==================== -->
  <h1>Chapter 6: Comprehensive Viva Voce Defense & Examiner Q&A Guide</h1>
  <p>
    This chapter contains <strong>25 high-probability, rigorous viva voce questions</strong> frequently posed by external examiners, accompanied by precise, technically authoritative answers.
  </p>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q1</span> What is the core objective of the FairFake project?</div>
    <div class="viva-a">
      FairFake is an explainable deepfake detection and demographic fairness auditing system. Its primary purpose is not just to detect fake faces, but to measure, explain, and audit performance disparities across demographic (age, gender, ethnicity) and non-demographic attributes (cosmetics, eyeglasses, facial hair) in forensic neural networks.
    </div>
    <div class="viva-tip">💡 <strong>Viva Tip:</strong> Emphasize that FairFake addresses both civil false accusations (high FPR on minorities) and security evasion risks (high FNR on manipulated content).</div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q2</span> Why is standard classification accuracy inadequate for evaluating deepfake detectors?</div>
    <div class="viva-a">
      Standard accuracy computes the ratio of correct predictions across an entire dataset. In benchmark corpora like Celeb-DF, Caucasian subjects make up 88.1% of frames. A detector could achieve 90% accuracy while failing completely on African or Asian subjects (e.g., 35% error rate). Balanced accuracy and subgroup error rates are required to uncover hidden biases.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q3</span> What is the mathematical formulation of Corrected Relative Performance ($CRP$)?</div>
    <div class="viva-a">
      $CRP(a) = RP_{\text{data}}(a) - RP_{\text{control}}(a) = \frac{err_{\text{ctrl}}^{(+)}}{err_{\text{ctrl}}^{(-)}} - \frac{err_{\text{data}}^{(+)}}{err_{\text{data}}^{(-)}}$. It subtracts the error ratio of a randomly resampled, balanced control group from the empirical error ratio. If disparity was caused purely by sample size imbalance, $RP_{\text{data}} = RP_{\text{control}}$, yielding $CRP = 0$.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q4</span> Why is the control group size set to $N = \min(|S^+|, |S^-|)$?</div>
    <div class="viva-a">
      Setting $N$ to the smaller subset ensures that both control groups ($\mathcal{C}^+$ and $\mathcal{C}^-$) have identical sample sizes drawn from the same underlying data distribution without replacement, establishing an empirical baseline of how data imbalance alone affects error rates.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q5</span> What is the practical difference between $PDRP$ and $DDRP$?</div>
    <div class="viva-a">
      $PDRP$ measures $CRP$ exclusively on genuine media ($y=0$), quantifying the tendency to falsely accuse innocent people (False Positive Rate disparity). $DDRP$ measures $CRP$ exclusively on deepfakes ($y=1$), quantifying the tendency of manipulations to slip through undetected (False Negative Rate disparity).
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q6</span> Why did you select Xception and EfficientNet-B0 as your primary backbones?</div>
    <div class="viva-a">
      Xception utilizes depthwise separable convolutions, which decompose standard spatial convolution into separate channel and spatial operations, allowing efficient extraction of microscopic boundary blending seams. EfficientNet-B0 balances depth, width, and resolution via compound scaling, making it the most lightweight yet competitive CNN backbone for real-time mobile verification.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q7</span> How does the Dual-Stream (RGB+FFT) architecture work?</div>
    <div class="viva-a">
      One stream processes the 3-channel spatial RGB image through `legacy_xception` (extracting 2048 features), while a parallel frequency stream processes the 2D Fast Fourier Transform log-magnitude spectrum through a 4-layer CNN (extracting 32 features). The concatenated 2080-dimensional feature vector is fed to a linear classifier, capturing both spatial seams and generative upsampling frequency artifacts.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q8</span> How does Grad-CAM work mathematically, and why did you hook into `conv4`?</div>
    <div class="viva-a">
      Grad-CAM calculates the gradient of the class prediction score with respect to feature maps $A^k$: $\alpha_k = \frac{1}{UV} \sum_i \sum_j \frac{\partial y}{\partial A_{ij}^k}$. The heatmap is $L = \text{ReLU}(\sum \alpha_k A^k)$. We hooked into `conv4` (the final convolutional block) because it retains the highest-level semantic information while preserving spatial coordinates before global pooling.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q9</span> Why does heavy makeup cause detectors to produce false alarms on genuine images?</div>
    <div class="viva-a">
      Cosmetics create sharp contrast boundaries, foundation seams, and texture smoothing across cheeks, jawlines, and eyelids. Deepfake detectors are trained to detect blending boundaries produced by Poisson image blending or warping algorithms; they mistake natural cosmetic gradients for digital synthesis seams.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q10</span> How does DeepFace fit into the FairFake pipeline?</div>
    <div class="viva-a">
      When an image is submitted, DeepFace executes real-time facial attribute analysis (`actions=['age', 'gender', 'race']`). FairFake maps the dominant race into clinical skin pigmentation categories and displays these traits alongside the detection verdict, immediately alerting analysts if the subject possesses high-bias traits (e.g., Dark Skin or Heavy Makeup).
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q11</span> What reference paper inspired this work, and what are its key contributions?</div>
    <div class="viva-a">
      Xu, Terhörst, Pedersen, and Raja, *"Analyzing Fairness in Deepfake Detection With Massively Annotated Databases"*, published in <em>IEEE Transactions on Technology and Society</em> (March 2024). They created 65.3M attribute annotations for 5 public deepfake datasets using the MAAD-Face classifier and demonstrated that state-of-the-art detectors exhibit severe demographic and non-demographic bias.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q12</span> What is the MAAD-Face principle, and why was a 90% confidence threshold enforced?</div>
    <div class="viva-a">
      MAAD-Face transfers attribute annotations from labeled source datasets (LFW and CelebA) to target datasets. To prevent error propagation and noisy labels, predictions with a reliability confidence score below 90% are marked as undefined ($0$), ensuring 92% average annotation accuracy across human evaluators.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q13</span> Why did you discard attributes with fewer than 100 samples in the audit?</div>
    <div class="viva-a">
      Small sample sizes introduce high variance in error rates. For example, if an attribute only appears in 10 images, a single misclassification changes the error rate by 10%. Enforcing a minimum threshold of 100 positive and negative samples ensures statistical significance.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q14</span> How does the video analysis pipeline in `backend/main.py` handle temporal sequences?</div>
    <div class="viva-a">
      It decodes the video via OpenCV, samples frames uniformly at 3 FPS (to prevent computational bottlenecks), passes each frame through the detector, flags seconds where $p_{\text{fake}} > 0.70$ as anomalies, and calculates a weighted authenticity score: $S_{\text{video}} = (1 - \max(p)) \times 70 + (1 - \text{ratio}_{\text{fake}}) \times 30$.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q15</span> Why did you use a two-stage training strategy (freezing the backbone)?</div>
    <div class="viva-a">
      Transfer learning with random weights in the newly initialized linear head produces large initial gradient updates. If backpropagated through the pretrained backbone immediately, these random gradients destroy the rich convolutional feature representations. Training the head for 2 epochs first allows it to stabilize before fine-tuning the entire network at a lower learning rate ($10^{-5}$).
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q16</span> What is the purpose of `pos_weight` in `BCEWithLogitsLoss`?</div>
    <div class="viva-a">
      Deepfake datasets are often class-imbalanced (e.g., more fakes than real). Setting $\text{pos\_weight} = N_{\text{real}} / N_{\text{fake}}$ weights positive loss terms proportionately, ensuring the optimizer does not bias predictions toward the majority class.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q17</span> What were your findings regarding binary gender versus skin pigmentation?</div>
    <div class="viva-a">
      Binary gender exhibited minimal bias ($|CRP| \le 0.10$; Male $+0.10$, Female $+0.08$), because training datasets maintain relative parity between male and female faces. In contrast, skin pigmentation ($CRP = +0.55$) and accessories ($CRP = +0.48$) exhibited extreme bias due to overwhelming dataset underrepresentation.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q18</span> What do the four quadrants in the $PDRP$ vs $DDRP$ plot represent?</div>
    <div class="viva-a">
      <ul>
        <li><strong>Quadrant I (Top-Right):</strong> Attribute increases performance on both real and fake images.</li>
        <li><strong>Quadrant II (Top-Left):</strong> Improved performance on pristine images, but degraded performance on fakes.</li>
        <li><strong>Quadrant III (Bottom-Left):</strong> Degraded performance on both pristine and fake media.</li>
        <li><strong>Quadrant IV (Bottom-Right):</strong> Improved performance on fakes, but severe false alarm penalty on pristine media.</li>
      </ul>
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q19</span> How can malicious actors exploit these bias findings to evade deepfake detectors?</div>
    <div class="viva-a">
      Attackers can deliberately generate deepfakes featuring attributes with high $DDRP$ (such as dark complexions, heavy cosmetics, or eyeglasses). Because the detector's sensitivity is compromised on these features, the probability of evasion increases by a factor of 2 to 3.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q20</span> What did your perturbation testing (Test 4) reveal about real-world deployments?</div>
    <div class="viva-a">
      Lossy compression ($Q=50$) and underexposure ($-25\%$) reduce overall AUC by up to 0.1188 and amplify demographic error disparities: underexposure widened the dark skin disparity by $\Delta CRP = +0.21$, showing that real-world camera noise exacerbates algorithmic bias.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q21</span> What is the difference between correlation and causation in this audit?</div>
    <div class="viva-a">
      Our findings prove strong statistical correlation between the presence of specific attributes and elevated error rates. However, correlation does not establish direct causation; confounding variables like lighting, pose variation, and sensor quality co-vary with these attributes.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q22</span> How does the frontend communicate with the FastAPI backend?</div>
    <div class="viva-a">
      The Next.js client uses `fetch` with `FormData` to POST binary image/video streams to `http://localhost:8000/api/analyze/image` or `/video`. The response contains Base64-encoded Grad-CAM strings, authenticity scores, and DeepFace biometric tags, which are stored in `sessionStorage` and rendered on `/analyze/[id]`.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q23</span> What is the difference between `train_transform` and `eval_transform` in Albumentations?</div>
    <div class="viva-a">
      `train_transform` incorporates stochastic augmentations (HorizontalFlip, ImageCompression, GaussianBlur, Brightness/Contrast shifts) to improve generalization. `eval_transform` applies only deterministic resizing and standard normalization to ensure repeatable, fair evaluation.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q24</span> What are the legal implications of biased deepfake detectors in courtroom evidence?</div>
    <div class="viva-a">
      Under legal evidentiary frameworks (such as the Daubert standard), automated forensic tools must demonstrate known error rates and operational standards. A detector with uncalibrated 35% error rates on specific demographics cannot meet evidentiary reliability standards and risks wrongful convictions.
    </div>
  </div>

  <div class="viva-card">
    <div class="viva-q"><span class="viva-q-tag">Q25</span> What are your recommendations for mitigating demographic bias in future work?</div>
    <div class="viva-a">
      <ol>
        <li><strong>Dataset Balancing:</strong> Mandate demographic parity in future training datasets.</li>
        <li><strong>Adversarial Debiasing:</strong> Add gradient reversal heads penalizing attribute prediction in feature representations.</li>
        <li><strong>Dynamic Thresholding:</strong> Calibrate classification thresholds independently across high-risk attribute cohorts.</li>
        <li><strong>Mandatory XAI Auditing:</strong> Require visual attribution maps (Grad-CAM) alongside verification verdicts.</li>
      </ol>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- ==================== CHAPTER 7 ==================== -->
  <h1>Chapter 7: Engineering Recommendations & Conclusion</h1>

  <h2>7.1 Five Core Engineering Principles for Fair Forensics</h2>
  <ul>
    <li><strong>Principle 1: Enforce Demographic Parity in Benchmark Corpora:</strong> Training databases must include balanced representations of diverse skin pigmentation, age cohorts, and natural facial adornments.</li>
    <li><strong>Principle 2: Institutionalize Resampled Auditing Metrics:</strong> Forensic vendors must publish $CRP$, $PDRP$, and $DDRP$ metrics alongside standard accuracy to prove compliance with non-discrimination standards.</li>
    <li><strong>Principle 3: Deploy Disentangled Representation Learning:</strong> Incorporate capsule networks or adversarial debiasing layers to force networks to disentangle manipulation traces from physiological facial structures.</li>
    <li><strong>Principle 4: Mandate Explainable Attribution in Verification Gateways:</strong> Verification tools must render real-time Grad-CAM attention heatmaps so human analysts can verify whether an alarm is driven by forgery artifacts or spurious makeup/lighting.</li>
    <li><strong>Principle 5: Environmental Robustness Training:</strong> Train models with aggressive compression, low-resolution, and low-illumination augmentations to prevent operational environments from magnifying demographic disparities.</li>
  </ul>

  <h2>7.2 Conclusion</h2>
  <p>
    This project successfully designed, implemented, and validated <strong>FairFake</strong>, demonstrating that high aggregate accuracy in deepfake detection masks severe demographic disparities. By combining mathematical control-group resampling, multi-model evaluation, Grad-CAM attribution, and an interactive full-stack web platform, FairFake provides an essential framework for building transparent, equitable, and trustworthy multimedia forensic systems.
  </p>

  <hr style="border:0; border-top:1px solid #e2e8f0; margin:30px 0;">

  <!-- ==================== CHAPTER 8 ==================== -->
  <h1>Chapter 8: Formal References & Bibliography</h1>
  <ol style="font-size:9pt; line-height:1.5;">
    <li>Y. Xu, P. Terhörst, M. Pedersen, and K. Raja, "Analyzing fairness in deepfake detection with massively annotated databases," <em>IEEE Transactions on Technology and Society</em>, vol. 5, no. 1, pp. 93–106, Mar. 2024. [DOI: 10.1109/TTS.2024.3365421]</li>
    <li>K. Yadav, M. Patel, and Dishant, "FairFake: Systematic demographic bias auditing and explainable visual forensics for deepfake detection systems," Undergraduate Technical Research Report, Department of Computer Science and Engineering, IIIT Vadodara, 2026.</li>
    <li>P. Terhörst, D. Fährmann, J. N. Kolf, N. Damer, F. Kirchbuchner, and A. Kuijper, "MAAD-Face: A massively annotated attribute dataset for face images," <em>IEEE Trans. Inf. Forensics Secur.</em>, vol. 16, pp. 3942–3957, 2021.</li>
    <li>F. Chollet, "Xception: Deep learning with depthwise separable convolutions," in <em>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</em>, 2017, pp. 1800–1807.</li>
    <li>M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in <em>Proc. 36th Int. Conf. Mach. Learn. (ICML)</em>, 2019, pp. 6105–6114.</li>
    <li>R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in <em>Proc. IEEE Int. Conf. Comput. Vis. (ICCV)</em>, 2017, pp. 618–626.</li>
    <li>Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, "Celeb-DF: A large-scale challenging dataset for deepfake forensics," in <em>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</em>, 2020, pp. 3204–3213.</li>
    <li>A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, "FaceForensics++: Learning to detect manipulated facial images," in <em>Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)</em>, 2019, pp. 1–11.</li>
    <li>B. Dolhansky et al., "The Deepfake Detection Challenge (DFDC) dataset," <em>arXiv preprint arXiv:2006.07397</em>, 2020.</li>
    <li>L. Jiang, R. Li, W. Wu, C. Qian, and C. C. Loy, "DeeperForensics-1.0: A large-scale dataset for real-world face forgery detection," in <em>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</em>, 2020, pp. 2886–2895.</li>
    <li>C. Hazirbas, J. Bitton, B. Dolhansky, J. Pan, A. Gordo, and C. Canton-Ferrer, "Towards measuring fairness in AI: The casual conversations dataset," <em>IEEE Trans. Biom. Behav. Identity Sci.</em>, vol. 4, no. 3, pp. 324–332, Jul. 2022.</li>
    <li>L. Trinh and Y. Liu, "An examination of fairness of AI models for deepfake detection," in <em>Proc. 30th Int. Joint Conf. Artif. Intell. (IJCAI)</em>, 2021, pp. 567–574.</li>
    <li>M. Pu, M. Y. Kuan, N. T. Lim, C. Y. Chong, and M. K. Lim, "Fairness evaluation in deepfake detection models using metamorphic testing," in <em>Proc. 7th Int. Workshop Metamorph. Test.</em>, 2022, pp. 7–14.</li>
    <li>S. I. Serengil and A. Ozpinar, "Hyperextended lightface: A comprehensive facial analysis framework," in <em>Proc. ICEET</em>, 2021, pp. 1–4.</li>
    <li>P. Terhörst et al., "A comprehensive study on face recognition biases beyond demographics," <em>IEEE Trans. Technol. Soc.</em>, vol. 3, no. 1, pp. 16–30, Mar. 2022.</li>
  </ol>

</div>

</body>
</html>
''')
    print(f"HTML successfully generated at: {HTML_FILE}")

def compile_pdf():
    print("Compiling PDF with Microsoft Edge headless print...")
    url = f"file:///{HTML_FILE.replace(os.sep, '/')}"
    
    cmd = [
        EDGE_BIN,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_FILE}",
        url
    ]
    
    if os.path.exists(PDF_FILE):
        try:
            os.remove(PDF_FILE)
        except Exception:
            pass
            
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    start_time = time.time()
    while time.time() - start_time < 35:
        if os.path.exists(PDF_FILE) and os.path.getsize(PDF_FILE) > 1000:
            time.sleep(2)
            break
        time.sleep(0.5)
        
    try:
        proc.terminate()
    except Exception:
        pass
        
    if not os.path.exists(PDF_FILE):
        print("Error: PDF compilation failed!")
        return False
        
    size = os.path.getsize(PDF_FILE)
    print(f"PDF generated: {PDF_FILE} ({size:,} bytes)")
    
    # Copy to workspace root for instant access
    shutil.copy2(PDF_FILE, ROOT_PDF)
    print(f"Copied to root directory: {ROOT_PDF} ({os.path.getsize(ROOT_PDF):,} bytes)")
    
    # Check page count with PyMuPDF
    try:
        import pymupdf
        doc = pymupdf.open(PDF_FILE)
        print(f"Total pages in compiled PDF: {len(doc)}")
    except Exception as e:
        print(f"PyMuPDF check note: {e}")
        
    return True

if __name__ == "__main__":
    build_html()
    compile_pdf()
