'''Plumbing shared by 01_allen_dataset.ipynb and 03_custom_dataset.ipynb: running commands, environment checks,
background training and plot display. Nothing here needs editing to use the notebooks.'''
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from celltransformer2_barseq.cli import example_path


def download_allen(root):
    '''Download the raw Allen 1 files into root/data/raw/ with the repository's checksummed downloader.'''
    script = example_path("allen") / "download.py"
    #stream the downloader's output into the notebook
    process = subprocess.Popen([sys.executable, "-u", str(script), "--root", str(root)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    if process.wait():
        raise RuntimeError(f"download failed ({process.returncode})")


def check_absolute(work_dir):
    '''Stop if work_dir is relative: commands run from inside it, so relative paths would point to the wrong place.'''
    if work_dir is not None and not Path(work_dir).is_absolute():
        raise ValueError(f"{work_dir} is a relative path; use absolute paths in Step 1 (e.g. Path(...).resolve())")


def run(command, *args, bin_dir=Path(sys.executable).parent, cwd=None, gpu=None, quiet=False):
    '''Run a celltransformer-* command from bin_dir and print its output.

    Paths under cwd are printed relative to it (the command itself still gets absolute paths).
    quiet=True skips printing the command, unless it fails.'''
    check_absolute(cwd)
    args = [str(a) for a in args]

    #display only: drop the cwd prefix from paths
    def short(text):
        return text.replace(f"{cwd}/", "") if cwd is not None else text

    shown_command = f"$ {command} " + short(" ".join(args))
    if not quiet:
        print(shown_command)
    #in case a GPU is given, hide the others from the command
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None
    result = subprocess.run([str(Path(bin_dir) / command), *args], capture_output=True, text=True, cwd=cwd, env=env)
    print(short(result.stdout[-5000:]).strip() + "\n")
    if result.returncode:
        if quiet:
            print(shown_command)
        print(short(result.stderr[-5000:]))
        raise RuntimeError(f"{command} failed ({result.returncode})")
    return result


def safe_section(name):
    '''Section name as celltransformer-preprocess writes it into filenames.'''
    return str(name).replace(".", "_").replace(" ", "_")


def check_environment(dev_bin, rapids_bin):
    '''List the visible GPUs and check that both environments can run their steps.'''
    #visible GPUs
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)

    #training environment: PyTorch must see the GPU
    torch_ok = subprocess.run([str(Path(dev_bin) / "python"), "-c",
                               "import torch; assert torch.cuda.is_available()"],
                              capture_output=True).returncode == 0
    if not torch_ok:
        raise RuntimeError("PyTorch cannot use the GPU; see Software setup step 3")
    print("training environment: PyTorch can use the GPU")

    #RAPIDS environment: smoothing and clustering commands, and a working cuML/cuDF
    for command in ["celltransformer-smooth", "celltransformer-cluster"]:
        if not (Path(rapids_bin) / command).exists():
            raise RuntimeError(f"{command} not found in RAPIDS_BIN; see Software setup step 4")
    rapids_ok = subprocess.run([str(Path(rapids_bin) / "python"), "-c", "import cuml, cudf"],
                               capture_output=True).returncode == 0
    if not rapids_ok:
        raise RuntimeError("RAPIDS environment is broken (often pyarrow was upgraded); "
                           "see Software setup step 4")
    print("RAPIDS environment: smoothing and clustering commands found, cuML and cuDF import")

    print("\nENVIRONMENT CHECK: PASSED")


def neighbor_counts(parquet_path, patch_size, n_sample=5000, seed=42):
    '''Neighbors inside the square window for a random sample of cells in one section.'''
    section_metadata = pd.read_parquet(parquet_path, columns=["x", "y"])
    coords = section_metadata[["x", "y"]].to_numpy()
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(coords), size=min(n_sample, len(coords)), replace=False)
    #square window = Chebyshev (p=inf) ball of radius patch_size / 2, minus the cell itself
    return cKDTree(coords).query_ball_point(coords[sample], patch_size / 2, p=np.inf, return_length=True) - 1


def training_running(work_dir):
    '''True if the training process started from work_dir is still alive.'''
    pid_file = Path(work_dir) / "train.pid"
    if not pid_file.exists():
        return False
    proc = Path("/proc") / pid_file.read_text().strip()
    try:
        state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
        cmdline = (proc / "cmdline").read_bytes()
    #a missing process is not running
    except FileNotFoundError:
        return False
    #nor is a finished one not yet collected by its parent (state Z), or, after a reboot,
    #an unrelated process that reuses the pid
    return state != "Z" and b"celltransformer-train" in cmdline


def finished_run(checkpoint_dir):
    '''Most recent run folder in checkpoint_dir whose training finished, or None.'''
    #celltransformer-train writes wandb_run.json only after training returns
    runs = sorted(d for d in Path(checkpoint_dir).iterdir() if (d / "wandb_run.json").exists())
    return runs[-1] if runs else None


def start_training(dev_bin, overrides, work_dir, log_path, checkpoint_dir, startup_wait=20):
    '''Start celltransformer-train in the background, unless it is running or has already finished.

    Waits up to startup_wait seconds so that errors at startup (e.g. a bad override) are reported here.'''
    check_absolute(work_dir)
    if training_running(work_dir):
        print("training is already running; follow it in Step 8")
        return
    run = finished_run(checkpoint_dir)
    if run is not None:
        print(f"training already finished in {run}; not starting a new run.\n"
              "To retrain, use a new WORK_DIR or move checkpoints/ aside.")
        return
    #offline W&B
    env = {**os.environ, "WANDB_MODE": "offline"}
    pid_file = Path(work_dir) / "train.pid"
    command = shlex.join([str(Path(dev_bin) / "celltransformer-train"), *overrides])
    #bash starts training in the background and exits, so training is re-parented to init:
    #it is no longer a child of the kernel and survives kernel restarts and shutdowns
    subprocess.run(["bash", "-c", f"nohup {command} > {shlex.quote(str(log_path))} 2>&1 < /dev/null & "
                                  f"echo $! > {shlex.quote(str(pid_file))}"],
                   cwd=work_dir, env=env, start_new_session=True, check=True)
    pid = pid_file.read_text().strip()

    #report a crash at startup here rather than in Step 8
    for _ in range(startup_wait):
        time.sleep(1)
        if not training_running(work_dir):
            lines = Path(log_path).read_text(errors="replace").replace("\r", "\n").splitlines()
            print("\n".join(line for line in lines[-30:] if line.strip()))
            raise RuntimeError(f"training exited during startup; full log: {log_path}")
    print(f"training started (PID {pid}); log: {log_path}")


def show_training(work_dir, log_path, checkpoint_dir, n_lines=15):
    '''Print whether training is alive, the finished run Step 9 will use, and the log or checkpoints.'''
    running = training_running(work_dir)
    finished = finished_run(checkpoint_dir)
    print("training running:", running)
    print("finished run:", finished if finished is not None else "none yet")

    #the log is about the current or last failed attempt, so show it only while there is no finished run to use
    if running or finished is None:
        lines = Path(log_path).read_text(errors="replace").replace("\r", "\n").splitlines()
        print("\n".join(line for line in lines[-n_lines:] if line.strip()))

    #checkpoints of the run in progress or the finished run, else of the newest attempt
    runs = sorted(d for d in Path(checkpoint_dir).iterdir() if d.is_dir())
    shown = finished if finished is not None and not running else (runs[-1] if runs else None)
    if shown is not None:
        print(f"\nrun {shown.name}:")
        for path in sorted(shown.rglob("*.ckpt")):
            print("  ", path.relative_to(shown))


def show_domain_plots(domains_dir, tags, ks, n_show=3, width=500):
    '''Display the first n_show section plots for each embedding set and k.'''
    import io
    from IPython.display import Image, display
    from PIL import Image as PILImage

    def preview(path):
        #downscaled jpeg copy, so the notebook stores ~0.3 MB per plot instead of the ~7 MB png
        image = PILImage.open(path).convert("RGB")
        image.thumbnail((2 * width, 2 * width * image.height // image.width))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
        return Image(data=buffer.getvalue(), format="jpeg", width=width)

    for tag in tags:
        for k in ks:
            plot_dir = Path(domains_dir) / f"{tag}_cluster_plots_k{k}"
            plots = sorted(plot_dir.glob("*.png"))
            print(f"{tag}, k = {k}: showing {min(n_show, len(plots))} of {len(plots)} sections (all in {plot_dir})")
            for plot in plots[:n_show]:
                display(preview(plot))
