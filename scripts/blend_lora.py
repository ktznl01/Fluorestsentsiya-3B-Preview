# -*- coding: utf-8 -*-
"""LoRA 适配器权重插值：out = (1-λ)·A + λ·B。

用法：
  python blend_lora.py --a train\\zh-lora-3b-v3\\step-220 --b train\\zh-lora-3b-v6fix\\step-120 --lam 0.5 --name v6blend50
"""
import argparse, json, os, shutil
from safetensors.torch import load_file, save_file
from paths import ROOT as ROOT_PATH

ROOT = str(ROOT_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    a = os.path.join(ROOT, args.a) if not os.path.isabs(args.a) else args.a
    b = os.path.join(ROOT, args.b) if not os.path.isabs(args.b) else args.b
    out = os.path.join(ROOT, "train", args.name)
    os.makedirs(out, exist_ok=True)

    sa = load_file(os.path.join(a, "adapter_model.safetensors"))
    sb = load_file(os.path.join(b, "adapter_model.safetensors"))
    assert set(sa) == set(sb), "adapter keys differ"
    blended = {k: sa[k] * (1.0 - args.lam) + sb[k] * args.lam for k in sa}
    save_file(blended, os.path.join(out, "adapter_model.safetensors"))
    shutil.copy2(os.path.join(a, "adapter_config.json"), os.path.join(out, "adapter_config.json"))
    print(f"blended lam={args.lam} -> {out}  tensors={len(blended)}")


if __name__ == "__main__":
    main()
