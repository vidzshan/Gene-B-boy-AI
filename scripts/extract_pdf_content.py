from pypdf import PdfReader
import os

output_file = "pdf_content_analysis.txt"

def read_pdf(path, label, f_out):
    f_out.write(f"\n--- {label} START ---\n")
    try:
        reader = PdfReader(path)
        number_of_pages = len(reader.pages)
        f_out.write(f"File: {path}\n")
        f_out.write(f"Pages: {number_of_pages}\n")
        
        # Capture first 20 pages of Sample for TOC/Intro/Structure
        limit = 20 if "SAMPLE" in label else None
        
        for i, page in enumerate(reader.pages):
            if limit and i >= limit:
                f_out.write(f"\n... (Truncating Sample at page {limit}) ...\n")
                break
            content = page.extract_text()
            if content:
                f_out.write(f"\n[Page {i+1}]\n")
                f_out.write(content)
                f_out.write("\n")
            else:
                f_out.write(f"\n[Page {i+1}] (No text found)\n")
                
    except Exception as e:
        f_out.write(f"Error reading {path}: {e}\n")
    f_out.write(f"\n--- {label} END ---\n")

base_dir = r"c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\Project_report"

with open(output_file, "w", encoding="utf-8") as f:
    read_pdf(os.path.join(base_dir, "Sample project report.pdf"), "SAMPLE REPORT", f)
    read_pdf(os.path.join(base_dir, "My project report.pdf"), "DRAFT REPORT", f)

print(f"Written extraction to {output_file}")
