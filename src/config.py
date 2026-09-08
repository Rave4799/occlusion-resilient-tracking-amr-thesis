"""
config.py — loads config.yaml and resolves all paths relative to BASE_DIR.

BASE_DIR is the repo root. It is resolved once, here, and never hardcoded
anywhere else in the codebase. Override via the BASE_DIR environment
variable if running from a different working directory (e.g. inside Colab
with the repo mounted from Drive).
"""
import os
import yaml
from pathlib import Path

# Repo root: this file lives in src/, so parent.parent is the repo root.
BASE_DIR = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parent.parent))


def load_config(config_path: Path = None) -> dict:
    """Load config.yaml and resolve every path under 'paths:' to an
    absolute path rooted at BASE_DIR. Returns the config dict with paths
    replaced by pathlib.Path objects.
    """
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve every entry under 'paths' to an absolute Path
    resolved_paths = {}
    for key, value in cfg.get("paths", {}).items():
        if key == "base_dir":
            continue
        resolved_paths[key] = BASE_DIR / value
    cfg["paths"] = resolved_paths

    return cfg


if __name__ == "__main__":
    # Quick sanity check: run `python src/config.py` to confirm paths resolve
    cfg = load_config()
    print(f"BASE_DIR: {BASE_DIR}")
    for k, v in cfg["paths"].items():
        exists = "✓" if v.exists() else "✗ (missing)"
        print(f"  {k}: {v}  [{exists}]")
