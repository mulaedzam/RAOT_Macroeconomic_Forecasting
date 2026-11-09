from __future__ import annotations
import json, platform, sys
from pathlib import Path

packages = {}
for name in ["numpy", "pandas", "matplotlib", "statsmodels", "torch"]:
    try:
        mod = __import__(name)
        packages[name] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        packages[name] = f"not available: {exc}"

snapshot = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
}
try:
    import torch
    snapshot["torch_cuda_available"] = bool(torch.cuda.is_available())
    snapshot["torch_cuda_version"] = torch.version.cuda
    snapshot["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
except Exception:
    pass

out = Path(__file__).resolve().parents[1] / "environment" / "environment_snapshot.json"
out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
print(out)
