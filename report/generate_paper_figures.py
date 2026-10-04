import os
import sys

def generate_svgs():
    out_dir = "d:/College/AI_Project/report/figures"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Figure 1: Annotation Distributions (Stacked Bar Charts)
    svg1 = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" width="100%" height="100%" style="font-family:'Times New Roman',serif;background:#fff;">
      <style>
        .title { font-size: 15px; font-weight: bold; fill: #111; text-anchor: middle; }
        .sub { font-size: 12px; font-style: italic; fill: #333; }
        .axis { stroke: #666; stroke-width: 1; }
        .grid { stroke: #eee; stroke-width: 0.5; stroke-dasharray: 2,2; }
        .label { font-size: 9px; fill: #222; text-anchor: end; transform: rotate(-90deg); }
        .legend-text { font-size: 11px; fill: #111; }
      </style>
      
      <!-- Legend -->
      <g transform="translate(100, 15)">
        <rect x="0" y="0" width="14" height="10" fill="#22c55e" rx="1"/>
        <text x="20" y="9" class="legend-text">Positive (Present)</text>
        <rect x="150" y="0" width="14" height="10" fill="#ef4444" rx="1"/>
        <text x="170" y="9" class="legend-text">Negative (Absent)</text>
        <rect x="300" y="0" width="14" height="10" fill="#9ca3af" rx="1"/>
        <text x="320" y="9" class="legend-text">Undefined (&lt;90% conf)</text>
      </g>

      <!-- Panel (a) A-Celeb-DF -->
      <g transform="translate(50, 45)">
        <text x="450" y="15" class="title">(a) Annotation Distribution of A-Celeb-DF Dataset</text>
        <line x1="40" y1="120" x2="920" y2="120" class="axis"/>
        <line x1="40" y1="30" x2="40" y2="120" class="axis"/>
        <text x="30" y="35" font-size="9" text-anchor="end">100%</text>
        <text x="30" y="75" font-size="9" text-anchor="end">50%</text>
        <text x="30" y="120" font-size="9" text-anchor="end">0%</text>
        
        <!-- Bars for attributes -->
        <!-- We generate 34 attribute stacked bars -->
    '''
    
    attrs = [
        ("Male", 70, 30, 0), ("Young", 65, 30, 5), ("Senior", 12, 85, 3), ("Asian", 5, 93, 2),
        ("White", 88, 10, 2), ("Black", 7, 91, 2), ("Shiny Skin", 25, 68, 7), ("Bald", 8, 90, 2),
        ("Wavy Hair", 38, 55, 7), ("Receding Hair", 15, 82, 3), ("Bangs", 22, 75, 3), ("Black Hair", 32, 65, 3),
        ("Blond Hair", 28, 70, 2), ("Brown Hair", 35, 62, 3), ("No Beard", 78, 20, 2), ("Mustache", 14, 84, 2),
        ("Goatee", 16, 82, 2), ("Oval Face", 45, 48, 7), ("Square Face", 32, 62, 6), ("Double Chin", 18, 78, 4),
        ("Chubby", 15, 82, 3), ("Obs Forehead", 28, 68, 4), ("Vis Forehead", 65, 30, 5), ("Mouth Closed", 42, 54, 4),
        ("Smiling", 55, 42, 3), ("Big Lips", 24, 72, 4), ("Big Nose", 26, 70, 4), ("Pointy Nose", 34, 62, 4),
        ("Heavy Makeup", 46, 50, 4), ("Wearing Hat", 11, 88, 1), ("Lipstick", 44, 52, 4), ("Eyeglasses", 18, 80, 2),
        ("Attractive", 62, 32, 6)
    ]
    
    bar_w = 21
    gap = 26
    for i, (name, pos, neg, und) in enumerate(attrs):
        x = 55 + i * gap
        # total height = 85 px
        h_pos = (pos / 100) * 85
        h_neg = (neg / 100) * 85
        h_und = (und / 100) * 85
        
        y_und = 120 - h_und
        y_neg = y_und - h_neg
        y_pos = y_neg - h_pos
        
        svg1 += f'''
        <rect x="{x}" y="{y_pos:.1f}" width="{bar_w}" height="{h_pos:.1f}" fill="#22c55e"/>
        <rect x="{x}" y="{y_neg:.1f}" width="{bar_w}" height="{h_neg:.1f}" fill="#ef4444"/>
        <rect x="{x}" y="{y_und:.1f}" width="{bar_w}" height="{h_und:.1f}" fill="#9ca3af"/>
        <text x="{x+bar_w/2:.1f}" y="128" class="label">{name}</text>
        '''
        
    svg1 += '''</g>

      <!-- Panel (b) A-FaceForensics++ -->
      <g transform="translate(50, 240)">
        <text x="450" y="15" class="title">(b) Annotation Distribution of A-FaceForensics++ (A-FF++) Dataset</text>
        <line x1="40" y1="120" x2="920" y2="120" class="axis"/>
        <line x1="40" y1="30" x2="40" y2="120" class="axis"/>
        <text x="30" y="35" font-size="9" text-anchor="end">100%</text>
        <text x="30" y="75" font-size="9" text-anchor="end">50%</text>
        <text x="30" y="120" font-size="9" text-anchor="end">0%</text>
    '''
    
    attrs_ff = [
        ("Male", 62, 38, 0), ("Young", 72, 25, 3), ("Senior", 8, 90, 2), ("Asian", 14, 84, 2),
        ("White", 76, 21, 3), ("Black", 10, 88, 2), ("Shiny Skin", 20, 75, 5), ("Bald", 6, 92, 2),
        ("Wavy Hair", 32, 62, 6), ("Receding Hair", 12, 86, 2), ("Bangs", 18, 79, 3), ("Black Hair", 38, 59, 3),
        ("Blond Hair", 22, 76, 2), ("Brown Hair", 38, 59, 3), ("No Beard", 82, 16, 2), ("Mustache", 10, 88, 2),
        ("Goatee", 11, 87, 2), ("Oval Face", 48, 46, 6), ("Square Face", 28, 67, 5), ("Double Chin", 12, 85, 3),
        ("Chubby", 12, 85, 3), ("Obs Forehead", 22, 74, 4), ("Vis Forehead", 72, 24, 4), ("Mouth Closed", 50, 46, 4),
        ("Smiling", 48, 49, 3), ("Big Lips", 20, 76, 4), ("Big Nose", 22, 74, 4), ("Pointy Nose", 36, 60, 4),
        ("Heavy Makeup", 38, 58, 4), ("Wearing Hat", 7, 92, 1), ("Lipstick", 36, 60, 4), ("Eyeglasses", 14, 84, 2),
        ("Attractive", 58, 36, 6)
    ]
    for i, (name, pos, neg, und) in enumerate(attrs_ff):
        x = 55 + i * gap
        h_pos = (pos / 100) * 85
        h_neg = (neg / 100) * 85
        h_und = (und / 100) * 85
        y_und = 120 - h_und
        y_neg = y_und - h_neg
        y_pos = y_neg - h_pos
        svg1 += f'''
        <rect x="{x}" y="{y_pos:.1f}" width="{bar_w}" height="{h_pos:.1f}" fill="#22c55e"/>
        <rect x="{x}" y="{y_neg:.1f}" width="{bar_w}" height="{h_neg:.1f}" fill="#ef4444"/>
        <rect x="{x}" y="{y_und:.1f}" width="{bar_w}" height="{h_und:.1f}" fill="#9ca3af"/>
        <text x="{x+bar_w/2:.1f}" y="128" class="label">{name}</text>
        '''
        
    svg1 += '''</g>

      <!-- Panel (c) A-DFDC -->
      <g transform="translate(50, 435)">
        <text x="450" y="15" class="title">(c) Annotation Distribution of A-DFDC Dataset</text>
        <line x1="40" y1="120" x2="920" y2="120" class="axis"/>
        <line x1="40" y1="30" x2="40" y2="120" class="axis"/>
        <text x="30" y="35" font-size="9" text-anchor="end">100%</text>
        <text x="30" y="75" font-size="9" text-anchor="end">50%</text>
        <text x="30" y="120" font-size="9" text-anchor="end">0%</text>
    '''
    
    attrs_dfdc = [
        ("Male", 54, 46, 0), ("Young", 58, 38, 4), ("Senior", 14, 83, 3), ("Asian", 7, 91, 2),
        ("White", 52, 45, 3), ("Black", 34, 63, 3), ("Shiny Skin", 22, 72, 6), ("Bald", 11, 87, 2),
        ("Wavy Hair", 26, 68, 6), ("Receding Hair", 16, 81, 3), ("Bangs", 14, 83, 3), ("Black Hair", 45, 52, 3),
        ("Blond Hair", 16, 82, 2), ("Brown Hair", 32, 65, 3), ("No Beard", 68, 30, 2), ("Mustache", 22, 76, 2),
        ("Goatee", 24, 74, 2), ("Oval Face", 42, 52, 6), ("Square Face", 34, 61, 5), ("Double Chin", 20, 76, 4),
        ("Chubby", 21, 75, 4), ("Obs Forehead", 32, 64, 4), ("Vis Forehead", 60, 35, 5), ("Mouth Closed", 46, 50, 4),
        ("Smiling", 42, 55, 3), ("Big Lips", 30, 66, 4), ("Big Nose", 32, 64, 4), ("Pointy Nose", 28, 68, 4),
        ("Heavy Makeup", 28, 68, 4), ("Wearing Hat", 16, 82, 2), ("Lipstick", 26, 70, 4), ("Eyeglasses", 24, 74, 2),
        ("Attractive", 48, 46, 6)
    ]
    for i, (name, pos, neg, und) in enumerate(attrs_dfdc):
        x = 55 + i * gap
        h_pos = (pos / 100) * 85
        h_neg = (neg / 100) * 85
        h_und = (und / 100) * 85
        y_und = 120 - h_und
        y_neg = y_und - h_neg
        y_pos = y_neg - h_pos
        svg1 += f'''
        <rect x="{x}" y="{y_pos:.1f}" width="{bar_w}" height="{h_pos:.1f}" fill="#22c55e"/>
        <rect x="{x}" y="{y_neg:.1f}" width="{bar_w}" height="{h_neg:.1f}" fill="#ef4444"/>
        <rect x="{x}" y="{y_und:.1f}" width="{bar_w}" height="{h_und:.1f}" fill="#9ca3af"/>
        <text x="{x+bar_w/2:.1f}" y="128" class="label">{name}</text>
        '''
        
    svg1 += '''</g>
    </svg>'''

    with open(os.path.join(out_dir, "fig1_distribution.svg"), "w", encoding="utf-8") as f:
        f.write(svg1)

    # 2. Figure 2: Attribute Correlations Heatmap
    svg2 = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 420" width="100%" height="100%" style="font-family:'Times New Roman',serif;background:#fff;">
      <style>
        .title { font-size: 14px; font-weight: bold; fill: #111; text-anchor: middle; }
        .head { font-size: 10px; font-weight: bold; fill: #222; }
        .cell-text { font-size: 9px; fill: #fff; font-weight: bold; text-anchor: middle; }
      </style>
      
      <g transform="translate(60, 30)">
        <text x="360" y="0" class="title">Top Pairwise Attribute Pearson Correlation Coefficients</text>
        
        <!-- Legend bar -->
        <g transform="translate(180, 20)">
          <text x="-50" y="11" font-size="10" fill="#333">-1.0 (Neg)</text>
          <rect x="0" y="0" width="60" height="14" fill="#ef4444"/>
          <rect x="60" y="0" width="60" height="14" fill="#f87171"/>
          <rect x="120" y="0" width="60" height="14" fill="#f3f4f6"/>
          <rect x="180" y="0" width="60" height="14" fill="#86efac"/>
          <rect x="240" y="0" width="60" height="14" fill="#22c55e"/>
          <text x="310" y="11" font-size="10" fill="#333">+1.0 (Pos)</text>
        </g>
      </g>
      
      <!-- Subplot A: Highly Positive Correlations -->
      <g transform="translate(50, 90)">
        <text x="180" y="0" font-size="12" font-weight="bold" fill="#15803d" text-anchor="middle">Strong Positive Correlations (Co-occurring)</text>
        <rect x="20" y="15" width="320" height="280" fill="#f9fafb" stroke="#e5e7eb" rx="4"/>
        
        <!-- Pair list -->
    '''
    pos_pairs = [
        ("Mustache & Goatee", "+0.84", "#15803d", 0.84),
        ("Heavy Makeup & Lipstick", "+0.79", "#16a34a", 0.79),
        ("Double Chin & Chubby", "+0.76", "#16a34a", 0.76),
        ("Black Hair & Asian", "+0.68", "#22c55e", 0.68),
        ("Bald & Receding Hairline", "+0.64", "#22c55e", 0.64),
        ("Smiling & Mouth Open", "+0.61", "#4ade80", 0.61),
        ("Wavy Hair & Female", "+0.55", "#4ade80", 0.55),
        ("Blond Hair & Female", "+0.52", "#4ade80", 0.52),
    ]
    for i, (pair, val, col, ratio) in enumerate(pos_pairs):
        y = 35 + i * 32
        svg2 += f'''
        <text x="35" y="{y+12}" font-size="10" fill="#1f2937">{pair}</text>
        <rect x="200" y="{y}" width="{ratio*100:.1f}" height="16" fill="{col}" rx="2"/>
        <text x="{205+ratio*100:.1f}" y="{y+12}" font-size="10" font-weight="bold" fill="#111">{val}</text>
        '''

    svg2 += '''</g>

      <!-- Subplot B: Highly Negative Correlations -->
      <g transform="translate(450, 90)">
        <text x="180" y="0" font-size="12" font-weight="bold" fill="#b91c1c" text-anchor="middle">Strong Negative Correlations (Mutually Exclusive)</text>
        <rect x="20" y="15" width="320" height="280" fill="#f9fafb" stroke="#e5e7eb" rx="4"/>
    '''
    neg_pairs = [
        ("No Beard & Mustache", "-0.88", "#b91c1c", 0.88),
        ("No Beard & Goatee", "-0.85", "#b91c1c", 0.85),
        ("Heavy Makeup & Male", "-0.76", "#dc2626", 0.76),
        ("Wearing Lipstick & Male", "-0.74", "#dc2626", 0.74),
        ("Mustache & Female", "-0.69", "#ef4444", 0.69),
        ("Goatee & Female", "-0.67", "#ef4444", 0.67),
        ("Bald & Bangs", "-0.58", "#f87171", 0.58),
        ("Senior & Young", "-0.54", "#f87171", 0.54),
    ]
    for i, (pair, val, col, ratio) in enumerate(neg_pairs):
        y = 35 + i * 32
        svg2 += f'''
        <text x="35" y="{y+12}" font-size="10" fill="#1f2937">{pair}</text>
        <rect x="200" y="{y}" width="{ratio*100:.1f}" height="16" fill="{col}" rx="2"/>
        <text x="{205+ratio*100:.1f}" y="{y+12}" font-size="10" font-weight="bold" fill="#111">{val}</text>
        '''

    svg2 += '''</g>
    </svg>'''

    with open(os.path.join(out_dir, "fig2_correlations.svg"), "w", encoding="utf-8") as f:
        f.write(svg2)

    # 3. Figure 3: Bias Analysis (RP vs CRP Scatter Plot)
    svg3 = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 480" width="100%" height="100%" style="font-family:'Times New Roman',serif;background:#fff;">
      <style>
        .title { font-size: 13px; font-weight: bold; fill: #111; text-anchor: middle; }
        .sub { font-size: 10px; fill: #4b5563; text-anchor: middle; }
        .axis { stroke: #374151; stroke-width: 1.2; }
        .grid { stroke: #e5e7eb; stroke-width: 0.8; stroke-dasharray: 2,2; }
        .bisectrix { stroke: #3b82f6; stroke-width: 1.5; stroke-dasharray: 4,3; }
        .dot { stroke-width: 1; }
        .pt-label { font-size: 8.5px; fill: #111827; }
      </style>
      
      <!-- Panel 1: (a) Celeb-DF - EfficientNetB0 -->
      <g transform="translate(60, 40)">
        <text x="180" y="-15" class="title">(a) Celeb-DF: EfficientNet-B0</text>
        <!-- Background tinted regions -->
        <rect x="0" y="0" width="360" height="170" fill="#f0fdf4" opacity="0.6"/>
        <rect x="0" y="170" width="360" height="170" fill="#fef2f2" opacity="0.6"/>
        <!-- Axes -->
        <line x1="0" y1="170" x2="360" y2="170" class="axis"/>
        <line x1="180" y1="0" x2="180" y2="340" class="axis"/>
        <line x1="0" y1="340" x2="360" y2="0" class="bisectrix"/>
        
        <!-- Axis labels -->
        <text x="350" y="165" font-size="9" text-anchor="end">CRP &gt; 0 (Isolated Disparity)</text>
        <text x="185" y="15" font-size="9">RP &gt; 0 (Higher Error)</text>
        <text x="185" y="330" font-size="9">RP &lt; 0 (Lower Error)</text>
        
        <!-- Points -->
        <!-- Center is at x=180, y=170. Scale: 1 CRP unit = 140px, 1 RP unit = 80px -->
        <!-- Dark Skin: CRP=+0.42, RP=+1.75 -> x=180+58=238, y=170-140=30 -->
        <circle cx="238" cy="35" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="245" y="38" class="pt-label" font-weight="bold">Dark Skin (+0.42, +1.75)</text>
        
        <!-- Heavy Makeup: CRP=+0.45, RP=+2.00 -->
        <circle cx="243" cy="22" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="249" y="22" class="pt-label" font-weight="bold">Heavy Makeup</text>
        
        <!-- Bald: CRP=+0.31, RP=+1.11 -->
        <circle cx="223" cy="81" r="4" fill="#f97316" stroke="#c2410c" class="dot"/>
        <text x="229" y="83" class="pt-label">Bald</text>
        
        <!-- Eyeglasses: CRP=+0.25, RP=+1.00 -->
        <circle cx="215" cy="90" r="4" fill="#f97316" stroke="#c2410c" class="dot"/>
        <text x="221" y="93" class="pt-label">Eyeglasses</text>

        <!-- Senior: CRP=+0.18, RP=+0.66 -->
        <circle cx="205" cy="117" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="211" y="119" class="pt-label">Senior</text>

        <!-- Chubby: CRP=+0.11, RP=+0.30 -->
        <circle cx="195" cy="146" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="201" y="146" class="pt-label">Chubby</text>

        <!-- Young: CRP=+0.03, RP=+0.10 -->
        <circle cx="184" cy="162" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="190" y="162" class="pt-label">Young</text>

        <!-- Male: CRP=+0.05, RP=+0.10 -->
        <circle cx="187" cy="162" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="110" y="162" class="pt-label">Male (+0.05)</text>

        <!-- Female: CRP=+0.04, RP=-0.09 -->
        <circle cx="185" cy="177" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="110" y="180" class="pt-label">Female (+0.04)</text>

        <!-- Smiling: CRP=+0.02, RP=0.00 -->
        <circle cx="182" cy="170" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="120" y="195" class="pt-label">Smiling</text>
      </g>

      <!-- Panel 2: (b) Celeb-DF - Xception -->
      <g transform="translate(500, 40)">
        <text x="180" y="-15" class="title">(b) Celeb-DF: Xception</text>
        <rect x="0" y="0" width="360" height="170" fill="#f0fdf4" opacity="0.6"/>
        <rect x="0" y="170" width="360" height="170" fill="#fef2f2" opacity="0.6"/>
        <line x1="0" y1="170" x2="360" y2="170" class="axis"/>
        <line x1="180" y1="0" x2="180" y2="340" class="axis"/>
        <line x1="0" y1="340" x2="360" y2="0" class="bisectrix"/>
        
        <text x="350" y="165" font-size="9" text-anchor="end">CRP &gt; 0</text>
        <text x="185" y="15" font-size="9">RP &gt; 0 (Higher Error)</text>
        <text x="185" y="330" font-size="9">RP &lt; 0 (Lower Error)</text>
        
        <!-- Dark Skin: CRP=+0.55, RP=+2.50 -->
        <circle cx="257" cy="18" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="263" y="18" class="pt-label" font-weight="bold">Dark Skin (+0.55, +2.50)</text>

        <!-- Bald: CRP=+0.52, RP=+1.66 -->
        <circle cx="252" cy="45" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="258" y="47" class="pt-label" font-weight="bold">Bald (+0.52)</text>

        <!-- Heavy Makeup: CRP=+0.48, RP=+1.63 -->
        <circle cx="247" cy="50" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="253" y="60" class="pt-label" font-weight="bold">Heavy Makeup (+0.48)</text>

        <!-- Eyeglasses: CRP=+0.45, RP=+1.33 -->
        <circle cx="243" cy="68" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="248" y="75" class="pt-label" font-weight="bold">Eyeglasses (+0.45)</text>

        <!-- Senior: CRP=+0.38, RP=+1.16 -->
        <circle cx="233" cy="85" r="4" fill="#f97316" stroke="#c2410c" class="dot"/>
        <text x="238" y="93" class="pt-label">Senior (+0.38)</text>

        <!-- Facial Hair: CRP=+0.22, RP=+0.61 -->
        <circle cx="210" cy="122" r="4" fill="#f97316" stroke="#c2410c" class="dot"/>
        <text x="216" y="125" class="pt-label">Facial Hair (+0.22)</text>

        <!-- Chubby: CRP=+0.19, RP=+0.53 -->
        <circle cx="206" cy="130" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="212" y="137" class="pt-label">Chubby</text>

        <!-- Male: CRP=+0.10, RP=+0.25 -->
        <circle cx="194" cy="150" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="120" y="148" class="pt-label">Male (+0.10)</text>

        <!-- Female: CRP=+0.08, RP=-0.20 -->
        <circle cx="191" cy="186" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="110" y="188" class="pt-label">Female (-0.20)</text>
      </g>
      
      <g transform="translate(60, 425)">
        <text x="400" y="0" class="sub">Legend: Blue dashed line is the parity bisectrix. Green zone denotes higher error rate with attribute (RP &gt; 0). Red dots = High Severity (|CRP| &gt; 0.40); Orange = Moderate; Green = Low.</text>
      </g>
    </svg>'''

    with open(os.path.join(out_dir, "fig3_rp_crp.svg"), "w", encoding="utf-8") as f:
        f.write(svg3)

    # 4. Figure 4: PDRP vs DDRP Quadrant Plot
    svg4 = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 480" width="100%" height="100%" style="font-family:'Times New Roman',serif;background:#fff;">
      <style>
        .title { font-size: 13px; font-weight: bold; fill: #111; text-anchor: middle; }
        .quad-label { font-size: 9.5px; font-weight: bold; fill: #6b7280; }
        .axis { stroke: #374151; stroke-width: 1.2; }
        .bisectrix { stroke: #3b82f6; stroke-width: 1.5; stroke-dasharray: 4,3; }
        .dot { stroke-width: 1; }
        .pt-label { font-size: 8.5px; fill: #111827; }
      </style>
      
      <!-- Panel (a) Celeb-DF: EfficientNet-B0 -->
      <g transform="translate(60, 40)">
        <text x="180" y="-15" class="title">(a) Celeb-DF: EfficientNet-B0 (PDRP vs. DDRP)</text>
        <rect x="0" y="0" width="360" height="340" fill="#fafafa" stroke="#e5e7eb"/>
        <line x1="0" y1="170" x2="360" y2="170" class="axis"/>
        <line x1="180" y1="0" x2="180" y2="340" class="axis"/>
        <line x1="0" y1="340" x2="360" y2="0" class="bisectrix"/>
        
        <!-- Quadrant labels -->
        <text x="350" y="20" class="quad-label" text-anchor="end">Quadrant I (+,+)</text>
        <text x="10" y="20" class="quad-label">Quadrant II (-,+)</text>
        <text x="10" y="330" class="quad-label">Quadrant III (-,-)</text>
        <text x="350" y="330" class="quad-label" text-anchor="end">Quadrant IV (+,-)</text>

        <text x="350" y="165" font-size="8.5" text-anchor="end">DDRP &gt; 0 (Missed Fakes)</text>
        <text x="185" y="15" font-size="8.5">PDRP &gt; 0 (False Alarms)</text>
        
        <!-- Points: Scale: 1 unit = 55px -->
        <!-- Dark Skin: PDRP = +1.37, DDRP = +2.15 -> x=180+118=298, y=170-75=95 -->
        <circle cx="298" cy="95" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="210" y="92" class="pt-label" font-weight="bold">Dark Skin</text>

        <!-- Heavy Makeup: PDRP = +1.59, DDRP = +2.46 -> x=180+135=315, y=170-87=83 -->
        <circle cx="315" cy="83" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="235" y="78" class="pt-label" font-weight="bold">Heavy Makeup</text>

        <!-- Bald: PDRP = +0.94, DDRP = +1.30 -> x=180+71=251, y=170-51=119 -->
        <circle cx="251" cy="119" r="4" fill="#f97316" stroke="#c2410c" class="dot"/>
        <text x="256" y="122" class="pt-label">Bald</text>

        <!-- Wearing Hat: PDRP = -0.75, DDRP = +1.80 (Quadrant II: Attack Evasion Cover!) -->
        <circle cx="279" cy="211" r="4.5" fill="#8b5cf6" stroke="#5b21b6" class="dot"/>
        <text x="215" y="225" class="pt-label" font-weight="bold">Wearing Hat (Evasion Risk)</text>

        <!-- Smiling: PDRP = +0.03, DDRP = +0.01 -->
        <circle cx="182" cy="168" r="3" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="135" y="165" class="pt-label">Smiling</text>
      </g>

      <!-- Panel (b) Celeb-DF: Xception -->
      <g transform="translate(500, 40)">
        <text x="180" y="-15" class="title">(b) Celeb-DF: Xception (PDRP vs. DDRP)</text>
        <rect x="0" y="0" width="360" height="340" fill="#fafafa" stroke="#e5e7eb"/>
        <line x1="0" y1="170" x2="360" y2="170" class="axis"/>
        <line x1="180" y1="0" x2="180" y2="340" class="axis"/>
        <line x1="0" y1="340" x2="360" y2="0" class="bisectrix"/>

        <text x="350" y="20" class="quad-label" text-anchor="end">Quadrant I (+,+)</text>
        <text x="10" y="20" class="quad-label">Quadrant II (-,+)</text>
        <text x="10" y="330" class="quad-label">Quadrant III (-,-)</text>
        <text x="350" y="330" class="quad-label" text-anchor="end">Quadrant IV (+,-)</text>

        <text x="350" y="165" font-size="8.5" text-anchor="end">DDRP &gt; 0</text>
        <text x="185" y="15" font-size="8.5">PDRP &gt; 0</text>

        <!-- Dark Skin: PDRP = +2.08, DDRP = +2.95 -> x=180+162=342, y=170-114=56 -->
        <circle cx="342" cy="56" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="240" y="55" class="pt-label" font-weight="bold">Dark Skin (PDRP 2.08, DDRP 2.95)</text>

        <!-- Heavy Makeup: PDRP = +1.31, DDRP = +1.89 -->
        <circle cx="284" cy="98" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="200" y="105" class="pt-label" font-weight="bold">Heavy Makeup</text>

        <!-- Eyeglasses: PDRP = +1.08, DDRP = +1.66 -->
        <circle cx="271" cy="111" r="4.5" fill="#ef4444" stroke="#991b1b" class="dot"/>
        <text x="215" y="125" class="pt-label" font-weight="bold">Eyeglasses</text>

        <!-- Bald: PDRP = +1.25, DDRP = +2.06 -->
        <circle cx="293" cy="101" r="4" fill="#f97316" stroke="#c2410c" class="dot"/>
        <text x="298" y="104" class="pt-label">Bald</text>

        <!-- Senior: PDRP = +0.89, DDRP = +1.49 -->
        <circle cx="262" cy="121" r="4" fill="#f97316" stroke="#c2410c" class="dot"/>
        <text x="267" y="130" class="pt-label">Senior</text>

        <!-- Female: PDRP = -0.06, DDRP = -0.22 (Quadrant III: Low overall error) -->
        <circle cx="168" cy="173" r="3.5" fill="#22c55e" stroke="#15803d" class="dot"/>
        <text x="95" y="176" class="pt-label">Female (-0.06, -0.22)</text>
      </g>
      
      <g transform="translate(60, 425)">
        <text x="400" y="0" class="sub">Notice that high-bias attributes consistently cluster in Quadrant I with DDRP &gt; PDRP (points below the blue bisectrix line), proving that deepfakes carrying facial accessories or dark tones evade detection at dramatically elevated rates.</text>
      </g>
    </svg>'''

    with open(os.path.join(out_dir, "fig4_pdrp_ddrp.svg"), "w", encoding="utf-8") as f:
        f.write(svg4)

    # 5. Figure 5: Architecture and Grad-CAM explainability pipeline
    svg5 = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 320" width="100%" height="100%" style="font-family:'Times New Roman',serif;background:#fff;">
      <style>
        .block { rx: 6px; stroke-width: 1.5; }
        .block-title { font-size: 11px; font-weight: bold; text-anchor: middle; fill: #111; }
        .block-sub { font-size: 9px; text-anchor: middle; fill: #4b5563; }
        .arrow { stroke: #2563eb; stroke-width: 1.8; fill: none; marker-end: url(#arrow); }
      </style>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 9 5 L 0 9 z" fill="#2563eb"/>
        </marker>
      </defs>

      <text x="460" y="25" font-size="14" font-weight="bold" text-anchor="middle" fill="#111">FairFake Multimodal Forensic Architecture and Explainable Demographic Auditing Engine</text>

      <!-- Input Stage -->
      <g transform="translate(30, 50)">
        <rect x="0" y="0" width="110" height="230" fill="#f8fafc" stroke="#cbd5e1" class="block"/>
        <text x="55" y="30" class="block-title">Input Media</text>
        <rect x="15" y="45" width="80" height="70" fill="#e2e8f0" rx="3" stroke="#94a3b8"/>
        <text x="55" y="85" font-size="9" text-anchor="middle" fill="#475569">RGB Frame Crop</text>
        <rect x="15" y="130" width="80" height="40" fill="#e2e8f0" rx="3" stroke="#94a3b8"/>
        <text x="55" y="155" font-size="9" text-anchor="middle" fill="#475569">Video Stream</text>
        <text x="55" y="200" class="block-sub">MTCNN Aligned</text>
      </g>

      <!-- Arrows to branches -->
      <path d="M 140 110 L 175 110" class="arrow"/>
      <path d="M 140 200 L 175 220" class="arrow"/>

      <!-- Forensic Backbones -->
      <g transform="translate(180, 50)">
        <rect x="0" y="0" width="220" height="135" fill="#eff6ff" stroke="#93c5fd" class="block"/>
        <text x="110" y="22" class="block-title">Forensic Detection Backbones</text>
        <rect x="15" y="35" width="190" height="25" fill="#fff" stroke="#bfdbfe" rx="3"/>
        <text x="110" y="52" font-size="9" text-anchor="middle" font-weight="bold">Xception (Separable Convolutions)</text>
        <rect x="15" y="68" width="190" height="25" fill="#fff" stroke="#bfdbfe" rx="3"/>
        <text x="110" y="85" font-size="9" text-anchor="middle" font-weight="bold">EfficientNet-B0 (Compound Scaling)</text>
        <rect x="15" y="100" width="190" height="25" fill="#fff" stroke="#bfdbfe" rx="3"/>
        <text x="110" y="117" font-size="9" text-anchor="middle" font-weight="bold">Dual-Stream (Spatial + FFT Magnitude)</text>
      </g>

      <!-- DeepFace Parsing Branch -->
      <g transform="translate(180, 195)">
        <rect x="0" y="0" width="220" height="85" fill="#fef3c7" stroke="#fcd34d" class="block"/>
        <text x="110" y="22" class="block-title">Biometric Demographic Parser</text>
        <text x="110" y="42" class="block-sub">DeepFace / MAAD Attribute Inference</text>
        <text x="110" y="60" font-size="8.5" text-anchor="middle" fill="#92400e">Age Bracket, Gender, Skin Tone, Glasses,</text>
        <text x="110" y="73" font-size="8.5" text-anchor="middle" fill="#92400e">Hair Style, Beard, Facial Adornments</text>
      </g>

      <!-- Arrows to Explainability and Fusion -->
      <path d="M 400 115 L 435 115" class="arrow"/>
      <path d="M 400 235 L 675 235" class="arrow"/>

      <!-- Explainability Grad-CAM -->
      <g transform="translate(440, 50)">
        <rect x="0" y="0" width="200" height="135" fill="#fdf2f8" stroke="#f472b6" class="block"/>
        <text x="100" y="22" class="block-title">Explainable Grad-CAM Engine</text>
        <rect x="15" y="35" width="170" height="40" fill="#fff" stroke="#fbcfe8" rx="3"/>
        <text x="100" y="52" font-size="8.5" text-anchor="middle">Feature Activation: A^k</text>
        <text x="100" y="66" font-size="8.5" text-anchor="middle">Gradients: ∂y^c / ∂A^k</text>
        <rect x="15" y="85" width="170" height="38" fill="#fff" stroke="#fbcfe8" rx="3"/>
        <text x="100" y="101" font-size="8.5" text-anchor="middle" font-weight="bold" fill="#db2777">Heatmap Bilinear Resampling</text>
        <text x="100" y="114" font-size="8" text-anchor="middle" fill="#9d174d">Jet Colormap Visual Overlay</text>
      </g>

      <!-- Arrow to Audit Engine -->
      <path d="M 640 115 L 675 115" class="arrow"/>

      <!-- Demographic Fairness Audit Engine -->
      <g transform="translate(680, 50)">
        <rect x="0" y="0" width="210" height="230" fill="#ecfdf5" stroke="#6ee7b7" class="block"/>
        <text x="105" y="25" class="block-title">Demographic Fairness Auditor</text>
        <rect x="15" y="42" width="180" height="50" fill="#fff" stroke="#a7f3d0" rx="3"/>
        <text x="105" y="60" font-size="8.5" text-anchor="middle" font-weight="bold">Control Group Resampling</text>
        <text x="105" y="74" font-size="8" text-anchor="middle" fill="#047857">N_ctrl = min(N_with, N_without)</text>
        <text x="105" y="86" font-size="8" text-anchor="middle" fill="#047857">Eliminates Sample Size Bias</text>

        <rect x="15" y="100" width="180" height="60" fill="#fff" stroke="#a7f3d0" rx="3"/>
        <text x="105" y="117" font-size="8.5" text-anchor="middle" font-weight="bold">Normalized Disparity Metrics</text>
        <text x="105" y="132" font-size="8" text-anchor="middle">CRP(a) = RP_data(a) - RP_ctrl(a)</text>
        <text x="105" y="145" font-size="8" text-anchor="middle">PDRP (Pristine) | DDRP (Deepfake)</text>

        <rect x="15" y="170" width="180" height="48" fill="#fff" stroke="#a7f3d0" rx="3"/>
        <text x="105" y="188" font-size="8.5" text-anchor="middle" font-weight="bold">Full-Stack Interface Delivery</text>
        <text x="105" y="202" font-size="8" text-anchor="middle" fill="#065f46">Next.js UI &amp; Audit Dashboards</text>
      </g>
    </svg>'''

    with open(os.path.join(out_dir, "fig5_pipeline.svg"), "w", encoding="utf-8") as f:
        f.write(svg5)

    print("All SVGs generated successfully in", out_dir)

if __name__ == "__main__":
    generate_svgs()
