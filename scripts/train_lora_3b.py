# -*- coding: utf-8 -*-
"""在 Qwen2.5-3B-Instruct 上做中文文学 LoRA（CPU / bf16 / 梯度检查点）。

与 train_lora.py 的区别：用 tokenizer 自带的 ChatML 模板，不硬编码 <｜User｜>；
数据换成续写对 + 迁移示范；不训练 think 块。

用法：
  python scripts\\train_lora_3b.py --steps 250 --seq 512
"""
import argparse, json, math, os, random, time
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from paths import ROOT, TRAIN, SFT, PROCESSED

p = argparse.ArgumentParser()
p.add_argument("--model", default=str(ROOT / "base" / "Qwen2.5-3B-Instruct"))
p.add_argument("--out", default=str(TRAIN / "adapter"))
p.add_argument("--seq", type=int, default=512)
p.add_argument("--steps", type=int, default=250)
p.add_argument("--lr", type=float, default=1.2e-4)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--dapt-per", type=int, default=2, help="每 N 步里 1 步 DAPT，其余 SFT")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--save-every", type=int, default=50)
p.add_argument("--data", nargs="+", default=[str(SFT / "sft.jsonl")])
p.add_argument("--corpus", default=str(PROCESSED / "corpus.txt"))
p.add_argument("--seed", type=int, default=20261031)
p.add_argument("--init-adapter", default="", help="从已有 LoRA 继续训练（适配器目录）")
args = p.parse_args()

torch.set_num_threads(args.threads)
random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

print("tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
EOS = tok.eos_token
print("eos", repr(EOS), "chat_template", "yes" if tok.chat_template else "no", flush=True)

print("tokenize corpus...", flush=True)
raw = open(args.corpus, encoding="utf-8").read()
parts = []
for i in range(0, len(raw), 40000):
    parts.append(tok(raw[i:i + 40000], add_special_tokens=False)["input_ids"])
ids = torch.tensor([t for p in parts for t in p], dtype=torch.long)
chunks = [ids[i:i + args.seq] for i in range(0, len(ids) - args.seq + 1, args.seq)]
print(f"corpus tokens={len(ids)} chunks={len(chunks)}", flush=True)


def build_sft(rec):
    msgs = [{"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + rec["assistant"] + EOS
    pi = tok(prompt, add_special_tokens=False)["input_ids"]
    fi = tok(full, add_special_tokens=False)["input_ids"]
    assert fi[:len(pi)] == pi, "prefix mismatch"
    fi = fi[:args.seq]
    labels = [-100] * min(len(pi), len(fi)) + fi[len(pi):]
    return fi, labels


sft = []
for path in args.data:
    for line in open(path, encoding="utf-8"):
        if line.strip():
            sft.append(json.loads(line))
sft_data = [build_sft(r) for r in sft]
print(f"sft examples={len(sft_data)} avg_tokens={sum(len(x[0]) for x in sft_data)/len(sft_data):.0f}", flush=True)

print("loading model bf16...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True, local_files_only=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 task_type="CAUSAL_LM")
if args.init_adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True,
                                      local_files_only=True)
    print("continued from adapter:", args.init_adapter, flush=True)
else:
    model = get_peft_model(model, cfg)
model.train()
trainable = [q for q in model.parameters() if q.requires_grad]
print(f"loaded {time.time()-t0:.1f}s trainable={sum(q.numel() for q in trainable)/1e6:.1f}M", flush=True)

opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8)
warm = max(5, args.steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))


def make_batch(kind):
    if kind == "dapt":
        c = chunks[random.randrange(len(chunks))]
        x = torch.tensor(c, dtype=torch.long).unsqueeze(0)
        return x, x.clone()
    fi, lb = sft_data[random.randrange(len(sft_data))]
    x = torch.tensor(fi, dtype=torch.long).unsqueeze(0)
    return x, torch.tensor(lb, dtype=torch.long).unsqueeze(0)


logpath = os.path.join(args.out, "train_log.csv")
new_log = not os.path.exists(logpath)
log = open(logpath, "a", encoding="utf-8")
if new_log:
    log.write("step,kind,loss,lr,sec\n")

acc, t_start = 0.0, time.time()
for step in range(1, args.steps + 1):
    kind = "dapt" if args.dapt_per <= 0 else ("dapt" if (step % args.dapt_per) != 0 else "sft")
    x, y = make_batch(kind)
    t = time.time()
    out = model(input_ids=x, labels=y)
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    dt = time.time() - t
    acc += loss.item()
    if step % 5 == 0:
        print(f"step {step}/{args.steps} {kind} loss={acc/5:.4f} lr={sched.get_last_lr()[0]:.2e} {dt:.1f}s", flush=True)
        log.write(f"{step},{kind},{acc/5:.4f},{sched.get_last_lr()[0]:.3e},{dt:.1f}\n"); log.flush()
        acc = 0.0
    if step % args.save_every == 0 or step == args.steps:
        model.save_pretrained(os.path.join(args.out, f"step-{step}"))
    if step % 50 == 0:
        print(f"[{time.time()-t_start:.0f}s elapsed]", flush=True)

model.save_pretrained(os.path.join(args.out, "final"))
print("done", flush=True)

