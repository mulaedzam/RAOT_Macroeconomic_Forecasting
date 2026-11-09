from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "18_brazil_model_data_preparation_FIXED_ORDER.py",
    "19_brazil_frozen_q0_o1_evaluation.py",
    "20_brazil_q0_o1_statistical_evaluation.py",
    "21_sa_common12_model_data_preparation.py",
    "22_sa_common12_frozen_q0_o1_evaluation.py",
    "23_sa_common12_q0_o1_statistical_evaluation.py",
    "24_cross_country_generalisability.py",
]

for name in SCRIPTS:
    path = ROOT / "src" / name
    print(f"\n=== Running {name} ===", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)
