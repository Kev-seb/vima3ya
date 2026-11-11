# NimbusPay Ticket Triage

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (loaded in 4-bit QLoRA)  
**Best result:** E2 — clean data, r=16, 4-bit QLoRA, 3 epochs — `exact_match = X.X%`, `mean_field_acc = X.X%` on dev (200 rows)

## Repo contents

| File / dir | Description |
|---|---|
| `notebook.ipynb` | Full Colab notebook with outputs. Runs top-to-bottom on a fresh T4 in <3h and reproduces the final adapter and test predictions. |
| `adapter/` | Final LoRA adapter in PEFT format (`adapter_config.json` + `adapter_model.safetensors`) |
| `report.md` | Audit findings, experiment table, error analysis, next steps |
| `cleaning_log.csv` | Every row dropped or fixed in `train.jsonl` |
| `predictions_test.jsonl` | Predictions for the 400 test tickets |

## How to load the adapter

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "adapter/", max_seq_length=512, load_in_4bit=True)
FastLanguageModel.for_inference(model)
```

## Inference

```python
import torch, json
msgs = [
    {"role": "system", "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."},
    {"role": "user",   "content": "<ticket text here>"},
]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

