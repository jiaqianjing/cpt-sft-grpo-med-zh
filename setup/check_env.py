"""Preflight check — verify hardware, packages, secrets, and connectivity before any run.

CALLING SPEC:
    python setup/check_env.py            # prints a report; exit 0 if all critical checks pass
    python setup/check_env.py --full     # also require the training stack (torch/vllm/lm-eval)

Side effects: reads .env (via configs.loader.load_secrets), queries nvidia-smi, pings HF.
Never prints secret values — only presence.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.loader import load_secrets  # noqa: E402

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[warn]"


def _pkg(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    try:
        return __import__(name).__version__  # type: ignore[attr-defined]
    except Exception:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also require the training stack")
    args = ap.parse_args()
    ok = True

    print("== GPUs ==")
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run(
            [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        n = len([ln for ln in out.splitlines() if ln.strip()])
        print(out or "(none)")
        print(f"{OK if n >= 1 else BAD} {n} GPU(s) visible")
        ok &= n >= 1
    else:
        print(f"{BAD} nvidia-smi not found"); ok = False

    print("\n== Secrets (.env) ==")
    present = load_secrets()
    for key in ("WANDB_API_KEY", "GEMINI_API_KEY"):
        mark = OK if present.get(key) else BAD
        print(f"{mark} {key} {'set' if present.get(key) else 'MISSING'}")
        ok &= present.get(key, False)
    for key in ("ANTHROPIC_API_KEY", "HF_TOKEN"):
        print(f"{OK if present.get(key) else WARN} {key} {'set' if present.get(key) else 'not set (optional)'}")

    print("\n== Data-env packages ==")
    for name in ("pydantic", "datasets", "transformers", "google", "wandb", "datasketch"):
        v = _pkg(name)
        print(f"{OK if v else BAD} {name} {v or 'MISSING'}")
        ok &= v is not None

    if args.full:
        print("\n== Training-stack packages ==")
        for name in ("torch", "trl", "vllm", "lm_eval", "deepspeed"):
            v = _pkg(name)
            print(f"{OK if v else BAD} {name} {v or 'MISSING'}")
            ok &= v is not None
        try:
            import torch  # noqa: PLC0415
            print(f"{OK if torch.cuda.is_available() else BAD} torch.cuda "
                  f"available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
            ok &= torch.cuda.is_available()
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} torch import failed: {e}"); ok = False

    print("\n== Connectivity ==")
    try:
        import urllib.request  # noqa: PLC0415
        code = urllib.request.urlopen("https://huggingface.co", timeout=8).status  # noqa: S310
        print(f"{OK if code == 200 else WARN} huggingface.co -> HTTP {code}")
    except Exception as e:  # noqa: BLE001
        print(f"{WARN} huggingface.co unreachable: {e}")

    print("\n" + ("ALL CRITICAL CHECKS PASSED" if ok else "SOME CRITICAL CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
