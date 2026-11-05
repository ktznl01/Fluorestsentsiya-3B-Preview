# -*- coding: utf-8 -*-
"""合并 LoRA → GGUF f16 → Q4_K_M / Q5_K_M（3B 版，CPU 可跑）。

用法：
  python scripts\\build_gguf_3b.py --adapter train\\zh-lora-3b\\final --name fluorestsentsiya-3b-preview
"""
import argparse, os, subprocess, sys, time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from paths import ROOT as ROOT_PATH

ROOT = str(ROOT_PATH)
PY = sys.executable
CONV = os.path.join(ROOT, "runtime", "src", "llama.cpp-b12018")
QUANT = os.path.join(ROOT, "runtime", "vulkan", "llama-quantize.exe")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.join(ROOT, "base", "Qwen2.5-3B-Instruct"))
ap.add_argument("--adapter", default=os.path.join(ROOT, "train", "adapter", "final"))
ap.add_argument("--name", default="fluorestsentsiya-3b-preview")
ap.add_argument("--scale", type=float, default=1.0, help="LoRA 强度（合并前缩放 lora_B）")
ap.add_argument("--quants", default="Q4_K_M")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

MERGED = os.path.join(ROOT, "models", "merged", args.name + "-bf16")
GGUF = os.path.join(ROOT, "models", "gguf")
os.makedirs(GGUF, exist_ok=True)

if args.force or not os.path.exists(os.path.join(MERGED, "config.json")):
    print("== merge adapter ==", flush=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True)
    if args.scale != 1.0:
        n = 0
        for name, mod in model.named_modules():
            if name.endswith("lora_B") or ".lora_B." in name or name.endswith("lora_B.default"):
                pass
        # 逐模块缩放：peft 的 lora_B 是一个 ModuleDict，里面按 adapter 名存 Linear
        for mod in model.modules():
            if hasattr(mod, "lora_B"):
                for key, lin in mod.lora_B.items():
                    lin.weight.data.mul_(args.scale)
                    n += 1
        print(f"scaled {n} lora_B modules by {args.scale}", flush=True)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, local_files_only=True).save_pretrained(MERGED)
    print(f"merged in {time.time()-t0:.0f}s -> {MERGED}", flush=True)

f16 = os.path.join(GGUF, args.name + "-f16.gguf")
if args.force or not os.path.exists(f16):
    print("== convert to gguf f16 ==", flush=True)
    r = subprocess.run([PY, os.path.join(CONV, "convert_hf_to_gguf.py"), MERGED,
                        "--outfile", f16, "--outtype", "f16"], cwd=CONV)
    if r.returncode:
        sys.exit("convert failed")

for q in [x.strip() for x in args.quants.split(",") if x.strip()]:
    out = os.path.join(GGUF, f"{args.name}-{q}.gguf")
    if os.path.exists(out) and not args.force:
        print("skip existing", out, flush=True)
        continue
    print(f"== quantize {q} ==", flush=True)
    r = subprocess.run([QUANT, f16, out, q])
    if r.returncode:
        sys.exit(f"quantize {q} failed")

for f in sorted(os.listdir(GGUF)):
    print(f"  {f:34s} {os.path.getsize(os.path.join(GGUF, f))/1e9:.2f} GB")

