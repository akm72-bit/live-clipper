"""One-time setup: let Whisper captions use an AMD GPU.

Whisper's engine (CTranslate2) only speaks CUDA, and AMD drivers don't. ZLUDA sits in the middle and
translates those CUDA calls to the AMD driver, which is what makes GPU captions possible on a Radeon.

    python setup_zluda.py           install/update ZLUDA into ./zluda, then check the whole chain
    python setup_zluda.py --check   report what is installed and what is missing, download nothing

ZLUDA also needs a HIP SDK (AMD's compute runtime, separate from the GPU driver) and a recent AMD
driver. `--check` tells you exactly which piece is missing.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, urllib.request, zipfile

from live_clipper import HERE, find_hip_sdk, find_zluda, gpu_names, has_gpu, log

RELEASES = "https://api.github.com/repos/vosen/ZLUDA/releases"
# ZLUDA's cuBLAS/cuDNN are wrappers around these, so a HIP SDK without them is useless to us
NEEDED = ["rocblas.dll", "libhipblaslt.dll"]


def newest_windows_release():
    """(tag, download url) of the newest Windows ZLUDA build."""
    req = urllib.request.Request(RELEASES, headers={"User-Agent": "live-clipper"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rels = json.load(r)
    for rel in rels:  # the API lists newest first
        for asset in rel.get("assets", []):
            if asset["name"].startswith("zluda-windows") and asset["name"].endswith(".zip"):
                return rel["tag_name"], asset["browser_download_url"]
    raise SystemExit("no Windows ZLUDA build found in the latest ZLUDA releases")


def install_zluda():
    tag, url = newest_windows_release()
    log(f"downloading ZLUDA {tag}")
    tmp = tempfile.mkdtemp(prefix="zluda-")
    try:
        zip_path = os.path.join(tmp, "zluda.zip")
        with urllib.request.urlopen(url, timeout=600) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f)  # ~30 MB, no progress bar needed
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        src = os.path.join(tmp, "zluda")  # the zip holds a single zluda/ folder
        dest = os.path.join(HERE, "zluda")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"ZLUDA {tag} installed in {dest}")


def cuda_check(zluda_dir, hip):
    """Run ZLUDA's own test app: it loads and initialises every CUDA library Whisper may need."""
    env = dict(os.environ, HIP_PATH=hip, PATH=os.path.join(hip, "bin") + os.pathsep + os.environ.get("PATH", ""))
    exe = os.path.join(zluda_dir, "cuda_check.exe")
    r = subprocess.run([exe], capture_output=True, text=True, timeout=300, env=env, cwd=zluda_dir)
    lines = [l.strip() for l in (r.stdout + r.stderr).splitlines() if ":" in l]
    ok = [l for l in lines if ": OK" in l]
    bad = [l for l in lines if ": OK" not in l and not l.startswith("nvapi")]  # nvapi is never implemented
    for l in sorted(lines):
        log("   ", l)
    return ok, bad


def whisper_gpu_check(zluda_dir, hip):
    """The check that matters: does Whisper's engine see the AMD card as a CUDA device?"""
    env = dict(os.environ, HIP_PATH=hip, ZLUDA_RUNNING="1",
               PATH=os.path.join(hip, "bin") + os.pathsep + os.environ.get("PATH", ""))
    probe = os.path.join(tempfile.mkdtemp(prefix="zc-"), "probe.py")  # a file, since the launcher mangles `python -c`
    with open(probe, "w") as f:
        f.write("import ctranslate2\nprint('CUDA devices visible to Whisper:', ctranslate2.get_cuda_device_count())\n")
    try:
        r = subprocess.run([os.path.join(zluda_dir, "zluda.exe"), "--", sys.executable, probe],
                           capture_output=True, text=True, timeout=600, env=env)
    finally:
        shutil.rmtree(os.path.dirname(probe), ignore_errors=True)
    out = (r.stdout + r.stderr).strip()
    log("   ", out.splitlines()[-1] if out else "no output")
    return "CUDA devices visible to Whisper: 1" in out


def main():
    ap = argparse.ArgumentParser(description="Set up ZLUDA so Whisper can use an AMD GPU.")
    ap.add_argument("--check", action="store_true", help="only report what is installed, download nothing")
    ap.add_argument("--force", action="store_true", help="reinstall ZLUDA even if ./zluda already exists")
    a = ap.parse_args()

    gpus = gpu_names()
    log("GPU:", ", ".join(gpus) if gpus else "none detected")
    if not (has_gpu("amd") or has_gpu("radeon")):
        log("No AMD GPU found: this setup is not needed, the clipper runs happily on the CPU (and on NVENC).")
        return 0
    if has_gpu("nvidia"):
        log("An NVIDIA GPU is present: Whisper will use CUDA directly, no ZLUDA needed.")
        return 0

    zluda_dir = find_zluda()
    if a.check:
        log("ZLUDA:", zluda_dir or "not installed (run `python setup_zluda.py`)")
    elif not zluda_dir or a.force or os.path.abspath(zluda_dir) != os.path.join(HERE, "zluda"):
        install_zluda()
        zluda_dir = find_zluda()

    hip = find_hip_sdk()
    if not hip:
        log("HIP SDK: NOT FOUND. ZLUDA cannot work without it.")
        log("  Install the AMD HIP SDK, or for cards the official SDK doesn't cover (Radeon RX 6000 is often")
        log("  gfx1032, which is unsupported) grab a nightly `therock-dist-windows-gfx103X-*.tar.gz` from")
        log("  https://therock-nightly-tarball.s3.amazonaws.com/index.html and set HIP_PATH to the folder you")
        log("  extract it into (it must contain bin/rocblas.dll).")
        return 1
    log("HIP SDK:", hip)
    missing = [d for d in NEEDED if not os.path.exists(os.path.join(hip, "bin", d))]
    if missing:
        log(f"HIP SDK is incomplete: {', '.join(missing)} not in {os.path.join(hip, 'bin')}")
        log("  The official SDK includes them; a partial/old install often doesn't. Reinstall it or point")
        log("  HIP_PATH at a complete one.")
        return 1

    if not zluda_dir:
        return 1
    ok, bad = cuda_check(zluda_dir, hip)
    log(f"ZLUDA libraries: {len(ok)} OK, {len(bad)} failed")
    if bad:
        log("Something is wrong with the HIP SDK above; fix that before the clipper can use the GPU.")
        return 1
    if not whisper_gpu_check(zluda_dir, hip):
        log("ZLUDA works, but Whisper's engine still can't see the card. Please open an issue with the output above.")
        return 1

    log("All good: run `python live_clipper.py CHANNEL` and Whisper will caption on your AMD GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
