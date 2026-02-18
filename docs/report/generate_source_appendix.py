
import os

# List of files to interpret as "The Core Codebase"
target_files = [
    "gbmc_engine.py",
    "sasaki_brain.py",
    "realtime_audio_engine.py",
    "blender_receiver.py",
    "RUN_SASAKI_LIVE.py",
    "audio_brace_loader.py",
    "calc_mean_pose.py",
    "SASAKI/Improvisation_SLM.py",
    "SASAKI/Sasaki_Control_Center.py",
    "SASAKI/Sasaki_Master_System.py",
    "SASAKI/__init__.py",
    "feeder/feeder_brace.py",
    "net/hpi_gcn.py", 
    "net/utils/graph.py",
]

base_dir = r"c:\Users\kos04\OneDrive\Desktop\vidz\GCN\HPI-GCN"
output_path = os.path.join(base_dir, "Project_report", "Chapter_13_Full_Source.md")

with open(output_path, "w", encoding="utf-8") as outfile:
    outfile.write("# Chapter 13: Complete Source Code Repository\n\n")
    outfile.write("This chapter contains the complete source code for the critical components of the system.\n\n")
    
    for fname in target_files:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            outfile.write(f"## File: {fname}\n")
            outfile.write(f"**Location:** `{fpath}`\n\n")
            outfile.write("```python\n")
            try:
                with open(fpath, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"# Error reading file: {e}\n")
            outfile.write("\n```\n\n")
            outfile.write("<div style='page-break-after: always;'></div>\n\n")
        else:
            print(f"Skipping {fname} (Not Found)")

print(f"Generated {output_path}")
