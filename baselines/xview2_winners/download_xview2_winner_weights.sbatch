#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=download_xview2_weights
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/download_xview2_weights_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/download_xview2_weights_%j.err

set -euo pipefail
set -x

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners
cd "$BASE"

mkdir -p xview2_first_place/weights
mkdir -p xview2_second_place/weights
mkdir -p xview2_third_place/weights

python - <<'PY'
import json
import time
import urllib.request
from pathlib import Path

RELEASES = [
    ("DIUx-xView/xView2_first_place", "final", "xview2_first_place/weights"),
    ("selimsef/xview2_solution", "0.0.1", "xview2_second_place/weights"),
    ("BloodAxe/xView2-Solution", "v1.0", "xview2_third_place/weights"),
]

def get_json(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "xview2-weight-downloader",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

def download(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "xview2-weight-downloader"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        total = r.headers.get("Content-Length")
        total = int(total) if total else None
        done = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"{dst.name}: {done * 100 / total:.1f}%", flush=True)
    print(f"Saved: {dst}", flush=True)

for repo, tag, out_dir in RELEASES:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    print("=" * 80, flush=True)
    print(f"Downloading release assets", flush=True)
    print(f"Repo: {repo}", flush=True)
    print(f"Tag: {tag}", flush=True)
    print(f"Output: {out}", flush=True)
    print("=" * 80, flush=True)

    release = get_json(api)
    assets = release.get("assets", [])

    if not assets:
        raise RuntimeError(f"No release assets found for {repo} tag {tag}")

    print(f"Found {len(assets)} assets", flush=True)

    for asset in assets:
        name = asset["name"]
        url = asset["browser_download_url"]
        dst = out / name

        if dst.exists() and dst.stat().st_size > 0:
            print(f"SKIP existing file: {dst}", flush=True)
            continue

        print(f"Downloading: {name}", flush=True)

        for attempt in range(3):
            try:
                download(url, dst)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {name}: {e}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(10)

print("All release assets downloaded.", flush=True)
PY

echo "===== Combine first-place split weights if present ====="
cd "$BASE/xview2_first_place/weights"

if ls split-weights-* >/dev/null 2>&1; then
  cat split-weights-* > weights.tar.gz
  mkdir -p extracted_weights
  tar -xzf weights.tar.gz -C extracted_weights || true
fi

cd "$BASE"

echo "===== Final check ====="
echo "1st-place weights:"
find xview2_first_place/weights -maxdepth 3 -type f | sort

echo "2nd-place weights:"
find xview2_second_place/weights -maxdepth 3 -type f | sort

echo "3rd-place weights:"
find xview2_third_place/weights -maxdepth 3 -type f | sort

echo "DONE."