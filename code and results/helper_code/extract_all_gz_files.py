import os
import gzip
import shutil

def extract_all_gz(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".gz"):
                gz_path = os.path.join(dirpath, filename)
                out_path = gz_path[:-3]  # remove .gz

                with gzip.open(gz_path, "rb") as f_in:
                    with open(out_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                print(f"Extracted: {gz_path}")

# Usage
extract_all_gz("ts_out_ct_4")