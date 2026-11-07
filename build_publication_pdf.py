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

if os.path.exists(out_pdf):
    try:
        os.remove(out_pdf)
    except Exception:
        pass

print("Launching Edge print...")
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Poll for file update and process exit
start_time = time.time()
while time.time() - start_time < 35:
    if os.path.exists(out_pdf) and os.path.getsize(out_pdf) > 1000:
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
try:
    with open(out_pdf, 'rb') as fsrc, open(root_pdf, 'wb') as fdst:
        fdst.write(fsrc.read())
    print(f"Copied to root PDF: {root_pdf} ({os.path.getsize(root_pdf)} bytes)")
except Exception as e:
    subprocess.run(["powershell", "-Command", f'Copy-Item -Force "{out_pdf}" "{root_pdf}"'])
    print(f"Copied via powershell: {root_pdf}")

# Verify and render sample pages with pymupdf
doc = pymupdf.open(out_pdf)
print(f"Total PDF pages: {len(doc)}")
doc[2].get_pixmap(dpi=150).save(r"report\page3_formulas_verified.png")
doc[-2].get_pixmap(dpi=150).save(r"report\page_second_last_verified.png")
doc[-1].get_pixmap(dpi=150).save(r"report\page_last_verified.png")
print("Rendered verification images successfully!")
