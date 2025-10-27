import subprocess
import os
import comet_ml

current_dir = os.path.dirname(os.path.abspath(__file__))
comet_dir = os.path.join(current_dir, ".cometml-runs")

# Ensure all zip files exist
zip_files = [f for f in os.listdir(comet_dir) if f.endswith(".zip")]


for zip_file in zip_files:
    zip_path = os.path.join(comet_dir, zip_file)
    try:
        subprocess.run(["comet", "upload", zip_path], check=True)
    except:
        print(f"Failed to upload {zip_file}. Most probably experiment is already uploaded. Check the website for more info!")