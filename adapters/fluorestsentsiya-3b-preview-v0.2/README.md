# Fluorestsentsiya-3B-Preview v0.2 — PEFT adapter

- Base: `Qwen/Qwen2.5-3B-Instruct`
- LoRA: r=16, alpha=32, dropout=0.05, 29.9M trainable parameters
- Targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Training: continued from the v0.1 register adapter; 2k human-continuation SFT with offline teacher revisions
- Format: PEFT `adapter_model.safetensors` + `adapter_config.json`

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "adapters/fluorestsentsiya-3b-preview-v0.2")
tok = AutoTokenizer.from_pretrained(base)
```

The adapter config stores the base model id; pass the local base model explicitly if you run offline.
Weights are distributed as release assets and are not tracked in git.
