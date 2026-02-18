
import os

draft_path = r"c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\Project_report\Project_Design_III_Report_Draft.md"
sample_path = r"c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN\pdf_content_analysis.txt"

def count_words(text):
    return len(text.split())

if os.path.exists(draft_path):
    with open(draft_path, 'r', encoding='utf-8') as f:
        draft_content = f.read()
    draft_count = count_words(draft_content)
    print(f"Draft Report Word Count: {draft_count}")
else:
    print("Draft report not found.")

if os.path.exists(sample_path):
    with open(sample_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # The sample report is the first part. Let's find the split point if possible.
        # Based on previous turns, the file contains "Sample project report.pdf" content first.
        # We'll just count the whole thing as a rough upper bound if we can't find a split, 
        # but better to try and find the transition to the draft if it exists.
        # Assuming the first half is the sample.
        # Let's just look for "My project report" which indicates the start of the second pdf.
        split_marker = "My project report.pdf"
        if split_marker in content:
            sample_content = content.split(split_marker)[0]
        else:
            sample_content = content # Fallback
            
    sample_count = count_words(sample_content)
    print(f"Sample Report Content (Extracted) Word Count: ~{sample_count}")
else:
    print("Sample analysis file not found.")
