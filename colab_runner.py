# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Privacy-Preserving FL over Noisy AIoT Links — Google Colab Runner  ║
# ║                                                                     ║
# ║  Instructions:                                                      ║
# ║  1. Open Google Colab: https://colab.research.google.com            ║
# ║  2. Create a new notebook                                           ║
# ║  3. Set runtime: Runtime → Change runtime type → T4 GPU             ║
# ║  4. Copy each section below into a separate Colab cell              ║
# ║     (split at the lines marked ── CELL ──)                          ║
# ║  5. Run cells in order                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 1: Verify GPU & Install Dependencies ──
# ══════════════════════════════════════════════════════════════════════════

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠ NO GPU — go to Runtime → Change runtime type → T4 GPU")

# Install packages
import subprocess
subprocess.run(["pip", "install", "-q", "opacus", "pyarrow", "tqdm"])
print("\n✓ Dependencies installed")


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 2: Mount Google Drive ──
# ══════════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')
print("\n✓ Google Drive mounted")
print("  Your files are at: /content/drive/MyDrive/")


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 3: Set File Paths ──
# ══════════════════════════════════════════════════════════════════════════

import os

# ╔═════════════════════════════════════════════════════════════════════╗
# ║  EDIT THESE THREE PATHS TO MATCH YOUR FILES                       ║
# ╚═════════════════════════════════════════════════════════════════════╝

SCRIPT = "/content/drive/MyDrive/fl_noisy_link.py"
DATASET = "/content/drive/MyDrive/merged_wisdm.csv"
OUTPUT  = "/content/fl_results"

# ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT, exist_ok=True)

for label, path in [("Script", SCRIPT), ("Dataset", DATASET)]:
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1e6
        print(f"  ✓ {label}: {path}  ({mb:.0f} MB)")
    else:
        print(f"  ✗ {label}: {path}  — FILE NOT FOUND")
        print(f"    → Check the path above and re-run this cell")


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 4: Convert CSV → Parquet (run once, skip if already done) ──
# ══════════════════════════════════════════════════════════════════════════

import time
import pandas as pd

if DATASET.endswith(".csv"):
    PARQUET = DATASET.rsplit(".", 1)[0] + ".parquet"

    if os.path.exists(PARQUET):
        mb = os.path.getsize(PARQUET) / 1e6
        print(f"✓ Parquet already exists: {PARQUET}  ({mb:.0f} MB)")
    else:
        print(f"Converting CSV → Parquet ...")
        t0 = time.time()

        df = pd.read_csv(DATASET, engine="c", low_memory=False)
        print(f"  Read {len(df):,} rows in {time.time()-t0:.1f}s")

        # Clean & optimise
        df.columns = df.columns.str.strip().str.lower().str.replace(";", "", regex=False)
        for col in ("x", "y", "z"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        for col in ("device", "sensor", "activity_code", "activity_label",
                    "subject_id", "user"):
            if col in df.columns:
                df[col] = df[col].astype("category")

        df.to_parquet(PARQUET, engine="pyarrow", compression="snappy", index=False)
        del df

        csv_mb = os.path.getsize(DATASET) / 1e6
        pq_mb = os.path.getsize(PARQUET) / 1e6
        print(f"\n  ✓ CSV: {csv_mb:.0f} MB → Parquet: {pq_mb:.0f} MB "
              f"({(1-pq_mb/csv_mb)*100:.0f}% smaller)")
        print(f"  Time: {time.time()-t0:.1f}s")

    DATASET = PARQUET
    print(f"\n  Using: {DATASET}")
else:
    print(f"Already Parquet: {DATASET}")


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 5: Quick Sanity Check (~1-2 min) ──
# ══════════════════════════════════════════════════════════════════════════

print("Running 20-round sanity check ...\n")
os.system(
    f'python "{SCRIPT}" '
    f'--csv "{DATASET}" '
    f'--output "{OUTPUT}/sanity" '
    f'--rounds 20 --seeds 42 --clients 15 --no_early_stop'
)
print("\n" + "=" * 60)
print("  ✓ Sanity check done! Ready for full experiments.")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 6: Run ALL Experiments (~1-2 hours on T4) ──
#
#    OR skip this and use CELLS 7A-7F to run individually
# ══════════════════════════════════════════════════════════════════════════

import time
t_start = time.time()

os.system(
    f'python "{SCRIPT}" '
    f'--csv "{DATASET}" '
    f'--output "{OUTPUT}" '
    f'--all_experiments '
    f'--seeds 42,123,7,256,999 '
    f'--rounds 50 '
    f'--clients 30 '
    f'--schedule dynamic'
)

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  ✓ All experiments complete in {elapsed/3600:.1f} hours")
print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 7A: Schedule Comparison only (~6-10 min on T4) ──
# ══════════════════════════════════════════════════════════════════════════

os.system(
    f'python "{SCRIPT}" '
    f'--csv "{DATASET}" --output "{OUTPUT}" '
    f'--seeds 42,123,7,256,999 --rounds 50 --clients 30 '
    f'--compare_schedules'
)


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 7B: Privacy-Utility Curve only (~20-35 min on T4) ──
# ══════════════════════════════════════════════════════════════════════════

os.system(
    f'python "{SCRIPT}" '
    f'--csv "{DATASET}" --output "{OUTPUT}" '
    f'--seeds 42,123,7,256,999 --rounds 50 --clients 30 '
    f'--privacy_curve'
)


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 7C: Ablation Study only (~18-28 min on T4) ──
# ══════════════════════════════════════════════════════════════════════════

os.system(
    f'python "{SCRIPT}" '
    f'--csv "{DATASET}" --output "{OUTPUT}" '
    f'--seeds 42,123,7,256,999 --rounds 50 --clients 30 '
    f'--run_ablation'
)


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 7D: Channel Awareness only (~6-10 min on T4) ──
# ══════════════════════════════════════════════════════════════════════════

os.system(
    f'python "{SCRIPT}" '
    f'--csv "{DATASET}" --output "{OUTPUT}" '
    f'--seeds 42,123,7,256,999 --rounds 50 --clients 30 '
    f'--channel_awareness'
)


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 7E: Baselines (noisy vs reliable) only (~6-10 min on T4) ──
# ══════════════════════════════════════════════════════════════════════════

os.system(
    f'python "{SCRIPT}" '
    f'--csv "{DATASET}" --output "{OUTPUT}" '
    f'--seeds 42,123,7,256,999 --rounds 50 --clients 30 '
    f'--baselines'
)


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 7F: RDP Validation only (~30 sec) ──
# ══════════════════════════════════════════════════════════════════════════

os.system(
    f'python "{SCRIPT}" '
    f'--csv "{DATASET}" --output "{OUTPUT}" '
    f'--seeds 42,123,7,256,999 '
    f'--validate_rdp'
)


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 8: View Results ──
# ══════════════════════════════════════════════════════════════════════════

import pandas as pd
from IPython.display import display

print(f"\nFiles in {OUTPUT}/\n")
for f in sorted(os.listdir(OUTPUT)):
    if os.path.isfile(os.path.join(OUTPUT, f)):
        kb = os.path.getsize(os.path.join(OUTPUT, f)) / 1024
        print(f"  {f:<50s}  {kb:.1f} KB")

# Display result tables
tables = [
    ("exp1_schedules.csv",       "Exp 1: Schedule Comparison"),
    ("exp2_privacy_utility.csv", "Exp 2: Privacy-Utility Curve"),
    ("exp3_ablation.csv",        "Exp 3: Ablation Study"),
    ("exp4_channel.csv",         "Exp 4: Channel Awareness"),
    ("exp5_baselines.csv",       "Exp 5: Baselines"),
    ("rdp_validation.csv",       "RDP Validation"),
]

for fname, title in tables:
    path = os.path.join(OUTPUT, fname)
    if os.path.exists(path):
        print(f"\n── {title} ──")
        display(pd.read_csv(path))


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 9: View Figures ──
# ══════════════════════════════════════════════════════════════════════════

from IPython.display import Image, display
import glob

figs = sorted(glob.glob(os.path.join(OUTPUT, "fig*.png")))

if figs:
    for fig_path in figs:
        print(f"\n── {os.path.basename(fig_path)} ──")
        display(Image(filename=fig_path, width=700))
else:
    print("No figures yet — run experiments first.")


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 10: Download Results ──
# ══════════════════════════════════════════════════════════════════════════

import shutil
from google.colab import files

# Option A: Download as ZIP
zip_path = shutil.make_archive("/content/fl_results_download", "zip", OUTPUT)
print(f"ZIP: {zip_path}  ({os.path.getsize(zip_path)/1e6:.1f} MB)")
files.download(zip_path)


# ══════════════════════════════════════════════════════════════════════════
# ── CELL 11 (alternative): Save to Google Drive instead of downloading ──
# ══════════════════════════════════════════════════════════════════════════

DRIVE_DEST = "/content/drive/MyDrive/fl_results_final"

if os.path.exists("/content/drive/MyDrive"):
    if os.path.exists(DRIVE_DEST):
        shutil.rmtree(DRIVE_DEST)
    shutil.copytree(OUTPUT, DRIVE_DEST)
    print(f"✓ Copied to: {DRIVE_DEST}")
else:
    print("Drive not mounted — use ZIP download (Cell 10)")
