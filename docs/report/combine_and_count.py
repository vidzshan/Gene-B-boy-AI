
import os

filenames = [
    "Chapter_1_2_Expanded.md",
    "Chapter_3_Expanded.md",
    "Chapter_4_5_Expanded.md",
    "Chapter_6_7_8_Expanded.md",
    "Chapter_9_Appendix_Expanded.md",
    "Chapter_10_11_Manuals.md",
    "Chapter_12_AppendixB.md",
    "Chapter_13_Full_Source.md"
]

base_dir = r"c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\Project_report"
output_file = os.path.join(base_dir, "Project_Design_III_Report_Mega_Final.md")

combined_content = ""

# Add Title Page
title_page = """# Project Design III Project Report

**Academic Year:** Reiwa 7 (2025)  
**Department:** Information Engineering, Kanazawa Institute of Technology  
**Project Number:** 2EP082  
**Submission Date:** January 23, 2026  

---

## **Project Title**
**Transition-Aware Improvisational Dance Generation Algorithm**  
*(Japanese Title: トランジションを考慮した即興的なダンス生成アルゴリズム)*

---

## **Member List**
| Student ID | Name | Role |
| :--- | :--- | :--- |
| **4EP3-029** | **SASAKI Kosuke** | **Project Leader / Algorithm Architect** |
| **4EP4-067** | **KUDA UDAGE VIDUSHAN PRABASH** | **Systems Engineer / Implementation Lead** |

**Advisor:** Professor Tomohito Yamamoto

<div style="page-break-after: always;"></div>

"""
combined_content += title_page

for fname in filenames:
    fpath = os.path.join(base_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            combined_content += f.read() + "\n\n<div style='page-break-after: always;'></div>\n\n"
    else:
        print(f"Warning: {fname} not found")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(combined_content)

print(f"Combined report written to {output_file}")
word_count = len(combined_content.split())
print(f"Final Total Word Count: {word_count}")
