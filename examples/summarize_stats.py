import numpy as np
import os
import json
import subprocess
from collections import defaultdict


def main():
    results_dir = "examples/results/360_v2/3dgs_1m_compressed"
    scenes = ["garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room"]
    stage = "compress"

    summary = defaultdict(list)
    for scene in scenes:
        scene_dir = os.path.join(results_dir, scene)

        if stage == "compress":
            zip_path = f"{scene_dir}/compression.zip"
            if os.path.exists(zip_path):
                subprocess.run(f"rm {zip_path}", shell=True)
            subprocess.run(f"zip -r {zip_path} {scene_dir}/compression", shell=True)
            out = subprocess.run(
                f"stat -c%s {zip_path}", shell=True, capture_output=True
            )
            size = int(out.stdout)
            summary["size"].append(size)

        with open(os.path.join(scene_dir, f"stats/{stage}_step29999.json"), "r") as f:
            stats = json.load(f)
            for k, v in stats.items():
                summary[k].append(v)

    for k, v in summary.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()