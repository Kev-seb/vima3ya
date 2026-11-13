# NimbusPay Ticket Triage: LoRA fine-tune

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
**Loaded in 4-bit:** yes (nf4, double quantisation, fp16 compute). 
**Best result:** `E5_aug` (cleaned data + 800 rule-targeted augmented rows, QLoRA r=16, 3 epochs): dev exact match **91.5%**, mean field accuracy **98.2%**, 100% valid JSON.

| File | What it is |
|---|---|
| `notebook.ipynb` | Colab notebook with outputs. `MODE="final"` reproduces the adapter and test predictions in under 3 h on a T4 |
| `adapter/` | Final LoRA adapter in PEFT format |
| `predictions_test.jsonl` | Raw model outputs for the 400 test tickets |
| `report.md` | Audit, experiments, error analysis, next steps, AI-tool use |
| `cleaning_log.csv` | Every dropped or fixed train row (`id,action,reason`) |
| `audit.py`, `rules.py` | Part A: cleaning script and the rule-based reference extractor used to audit labels |
| `augment.py` | Experiment E5: rule-targeted synthetic tickets built from the cleaned train data |

## Loading the adapter (same as the notebook)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base_id)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, "adapter").eval()

msgs = [{"role": "system", "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."},
        {"role": "user", "content": "Received: 2026-08-14 (Fri)\nSource: email\n\n..."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(0)
out = model.generate(ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
```

Qwen2.5 supports a system role natively, so the system message is passed unchanged.
