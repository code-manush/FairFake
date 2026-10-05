import os
import time
import shutil
import subprocess
import pymupdf

edge_bin = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
html_file = os.path.abspath(r"report\paper.html")
out_pdf = os.path.abspath(r"report\FairFake_IEEE_Paper.pdf")
root_pdf = os.path.abspath(r"Analyzing_Fairness_in_Deepfake_Detection_With_Massively_Annotated_Databases.pdf")

url = f"file:///{html_file.replace(os.sep, '/')}"

cmd = [
    edge_bin,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--no-pdf-header-footer",
    f"--print-to-pdf={out_pdf}",
    url
]

print("Launching Edge print...")
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Poll for file update and process exit
start_time = time.time()
while time.time() - start_time < 30:
    if os.path.exists(out_pdf) and os.path.getmtime(out_pdf) > start_time - 1:
        # Give a small moment for file handle to close
        time.sleep(2)
        break
    time.sleep(0.5)

try:
    proc.terminate()
except Exception:
    pass

if not os.path.exists(out_pdf):
    print("Error: Output PDF was not created!")
    exit(1)

size = os.path.getsize(out_pdf)
print(f"FairFake_IEEE_Paper.pdf generated: {size} bytes")

# Copy to root PDF
shutil.copy2(out_pdf, root_pdf)
print(f"Copied to root PDF: {root_pdf} ({os.path.getsize(root_pdf)} bytes)")

# Verify and render sample pages with pymupdf
doc = pymupdf.open(out_pdf)
print(f"Total PDF pages: {len(doc)}")
doc[2].get_pixmap(dpi=150).save(r"report\page3_formulas_verified.png")
doc[-2].get_pixmap(dpi=150).save(r"report\page_second_last_verified.png")
doc[-1].get_pixmap(dpi=150).save(r"report\page_last_verified.png")
print("Rendered verification images successfully!")
