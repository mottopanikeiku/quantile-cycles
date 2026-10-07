"""Build the pinned proof on an ephemeral CPU-only Modal container.

Run from the repository root with Modal 1.5.3:
    modal run lean/modal_build.py
    modal run lean/modal_build.py --full
The pilot has a five-minute limit; --full has a thirty-minute limit.
No app is deployed and no GPU is requested.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
app = modal.App("quantile-cycles-lean")


def prepare(lakefile: str, toolchain: str) -> None:
    """Cache only the toolchain and pinned imported dependencies in the image."""
    import os

    os.environ["PATH"] = "/root/.elan/bin:" + os.environ["PATH"]
    os.environ["LEAN_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MATHLIB_NO_CACHE_ON_UPDATE"] = "1"
    subprocess.run(
        ["curl", "-fL", "-o", "/tmp/elan.tar.gz",
         "https://github.com/leanprover/elan/releases/download/v4.1.2/elan-x86_64-unknown-linux-gnu.tar.gz"],
        check=True, timeout=60,
    )
    subprocess.run(["tar", "-xzf", "/tmp/elan.tar.gz", "-C", "/tmp"], check=True)
    subprocess.run(["/tmp/elan-init", "-y", "--default-toolchain", "none"], check=True)
    work = Path("/proof")
    work.mkdir()
    (work / "lakefile.toml").write_text(lakefile)
    (work / "lean-toolchain").write_text(toolchain)
    subprocess.run(["lake", "update"], cwd=work, check=True, timeout=240)
    subprocess.run(
        ["lake", "exe", "cache", "get", "Mathlib.Data.Rat.Defs",
         "Mathlib.Tactic.NormNum", "Mathlib.Tactic.Linarith"],
        cwd=work, check=True, timeout=240,
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git", "zstd")
    .run_function(
        prepare,
        args=((HERE / "lakefile.toml").read_text(), (HERE / "lean-toolchain").read_text()),
        cpu=2, memory=2048, timeout=300,
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/bin:/usr/bin:/bin",
          "LEAN_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
)


def check(source: str, manifest: str | None) -> dict:
    work = Path("/proof")
    (work / "QuantileCycles.lean").write_text(source)
    actual_manifest = (work / "lake-manifest.json").read_text()
    if manifest is not None and json.loads(manifest) != json.loads(actual_manifest):
        raise RuntimeError("The committed manifest differs from the pinned image dependencies")
    logs = []
    for command in (["lake", "build"], ["lake", "env", "lean", "QuantileCycles.lean"]):
        result = subprocess.run(command, cwd=work, text=True, capture_output=True, timeout=280)
        logs.append({"command": command, "returncode": result.returncode,
                     "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            break
    return {"manifest": actual_manifest, "commands": logs,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "cpu_cores": 2, "memory_mib": 2048, "gpu": "none",
            "lean_version": subprocess.check_output(["lean", "--version"], text=True).strip()}


@app.function(image=image, cpu=2, memory=2048, timeout=300, max_containers=1)
def pilot(source: str, manifest: str | None) -> dict:
    return check(source, manifest)


@app.function(image=image, cpu=2, memory=2048, timeout=1800, max_containers=1)
def full(source: str, manifest: str | None) -> dict:
    return check(source, manifest)


@app.local_entrypoint()
def main(full: bool = False) -> None:
    manifest_path = HERE / "lake-manifest.json"
    result = (globals()["full"] if full else pilot).remote(
        (HERE / "QuantileCycles.lean").read_text(),
        manifest_path.read_text() if manifest_path.exists() else None,
    )
    manifest_path.write_text(result.pop("manifest"))
    (ROOT / "results" / "lean-build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if len(result["commands"]) != 2 or any(c["returncode"] for c in result["commands"]):
        raise RuntimeError("Lean rejected the proof; inspect results/lean-build.json")
    (ROOT / "results" / "lean-axioms.txt").write_text(result["commands"][1]["stdout"])
