"""Write train_umi.ipynb (the GPU-server notebook). Run once after editing the cells below."""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
cells = [
 (MD, """# UMI policy training (GPU server)

Bundle layout (this notebook's folder):
```
03_umi_policy_trainer/   trainer code + patches for Stanford UMI
data/v4.zarr.zip         dataset (+ v4.zarr.report.json sidecar, required)
runs/<run-id>/           output: checkpoints/, best.ckpt, manifest.json
```
Set `GPU` in the first cell, then run the cells top to bottom. Cells 1-3 are one-time setup; cell 4 trains (~1-2 h on a server GPU for 120 epochs)."""),
 (CODE, """import os
GPU = "2"   # physical GPU index from nvidia-smi; everything below sees only this one as cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
!nvidia-smi -i {GPU}
!python --version"""),
 (MD, "## 1. Python packages\nSkips the PyTorch install if a CUDA-enabled torch is already present."),
 (CODE, """import importlib.util, re, subprocess, sys
# V100 (sm_70) works with both wheels; the driver decides: CUDA 12.x driver -> cu126, 11.8+ -> cu118.
smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
driver_cuda = float(re.search(r"CUDA Version: (\\d+\\.\\d+)", smi).group(1))
assert driver_cuda >= 11.8, f"driver CUDA {driver_cuda} is too old for torch 2.6; ask the admin to update the driver"
index = "cu126" if driver_cuda >= 12.0 else "cu118"
need_torch = importlib.util.find_spec("torch") is None or not __import__("torch").cuda.is_available()
if need_torch:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch==2.6.0", "torchvision==0.21.0",
                           "--index-url", f"https://download.pytorch.org/whl/{index}"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "03_umi_policy_trainer/requirements.txt"])
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0),
                    f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")"""),
 (MD, "## 2. Stanford UMI checkout (pinned commit + local patches)"),
 (CODE, """%%bash
set -e
UMI=third_party/umi
[ -d $UMI/.git ] || git clone -q https://github.com/real-stanford/universal_manipulation_interface.git $UMI
git -C $UMI checkout -q d095ba9590df789df5189eea5ee7e431689038a6
sed -i "s/\\r$//" 03_umi_policy_trainer/patches/*.patch   # patches edited on Windows may carry CRLF
for p in $PWD/03_umi_policy_trainer/patches/*.patch; do   # absolute: git -C resolves relative paths inside the repo
  if git -C $UMI apply --check "$p" 2>/dev/null; then git -C $UMI apply "$p"; echo "applied $p";
  elif git -C $UMI apply --check -R "$p" 2>/dev/null; then echo "already applied $p";
  else echo "CANNOT APPLY $p"; exit 1; fi
done"""),
 (CODE, """import os
os.environ["UMI_ROOT"] = os.path.abspath("third_party/umi")
os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1\""""),
 (MD, "## 3. Dataset check"),
 (CODE, "!python 03_umi_policy_trainer/train_policy.py check data/v4.zarr.zip"),
 (MD, "## 4. Train\nChange `RUN_ID` / `EPOCHS` here. V100 16 GB: on CUDA out of memory, pass `--batch 16` and set `training.gradient_accumulate_every=2` in the profile. Re-running with the same `RUN_ID` resumes from the last checkpoint."),
 (CODE, """RUN_ID = "s22_v4_gpu"
EPOCHS = 120
!python 03_umi_policy_trainer/train_policy.py train data/v4.zarr.zip \
    --run-id {RUN_ID} --runs-dir runs --epochs {EPOCHS} \
    --profile 03_umi_policy_trainer/configs/policy_resnet18_gpu.yaml --eval-batch 32"""),
 (MD, "## 5. Result\n`best.ckpt` is chosen by validation position RMSE (mm). Compare `position_component_rmse_mm` with the `hold_current_pose` baseline: lower than baseline means the policy learned something."),
 (CODE, """import json
m = json.load(open(f"runs/{RUN_ID}/manifest.json"))
print(json.dumps({k: m[k] for k in m if k != "candidates"}, indent=2))"""),
 (CODE, """# Optional: pack the outputs to download (best.ckpt + manifest + config).
!cd runs/{RUN_ID} && tar czf ../{RUN_ID}_best.tgz best.ckpt manifest.json config.yaml dataset_check.json && ls -la ../{RUN_ID}_best.tgz"""),
]

def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == CODE:
        c.update(execution_count=None, outputs=[])
    return c

nb = {"cells": [cell(k, s) for k, s in cells], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("train_umi.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
