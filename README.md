# 🤖 Gemini Model Fine-Tuning Guide

> Fine-tune Google Gemini models on your own domain data using Vertex AI — turn a general-purpose model into a domain expert.

---

## 📋 Table of Contents
- [What is Fine-Tuning?](#what-is-fine-tuning)
- [Supported Models](#supported-models)
- [Pricing](#pricing)
- [Prerequisites](#prerequisites)
- [Dataset Format](#dataset-format)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Helper Functions](#helper-functions)
- [Launch Fine-Tuning Job](#launch-fine-tuning-job)
- [Use Your Fine-Tuned Model](#use-your-fine-tuned-model)
- [What Can You Build?](#what-can-you-build)

---

## 🧠 What is Fine-Tuning?

Fine-tuning = taking a pre-trained Gemini model and training it further on **your own domain data**.

Instead of describing what you want in every prompt, you **bake the knowledge directly into the model weights**.

```
Base Gemini Model  +  Your Domain Data  →  Domain Expert Model
```

**Example:**
| | Before Fine-Tuning | After Fine-Tuning |
|---|---|---|
| Question | "What's on your menu?" | "What's on your menu?" |
| Response | "I don't know your menu." | "We have Burgers, Fries, Combos from $8.99! 🍔" |

---

## 🤖 Supported Models

| Model | Fine-Tuning Type | Best For |
|---|---|---|
| `gemini-2.5-pro` | Supervised (SFT) | Complex reasoning, high accuracy |
| `gemini-2.5-flash` | SFT + Preference Tuning | Balanced performance & cost |
| `gemini-2.5-flash-lite` | SFT + Preference Tuning | High-volume, budget-friendly |

---

## 💰 Pricing

### Training Cost (one-time per run)
| Model | Training | Input Inference | Output Inference |
|---|---|---|---|
| `gemini-2.5-pro` | $25.00 / 1M tokens | $1.25 / 1M | $10.00 / 1M |
| `gemini-2.5-flash` | $5.00 / 1M tokens | $0.30 / 1M | $2.50 / 1M |
| `gemini-2.5-flash-lite` | $1.50 / 1M tokens | $0.10 / 1M | $0.40 / 1M |

> 💡 **Billed tokens** = dataset tokens × epochs (default: 3 epochs)

**Example cost estimate** — 1,000 examples × 500 tokens × 3 epochs = 1.5M tokens:
- Flash Lite → **$2.25** 🟢
- Flash → **$7.50** 🟡
- Pro → **$37.50** 🔴

---

## ✅ Prerequisites

```bash
# Install dependencies
pip install google-generativeai google-cloud-storage google-cloud-aiplatform

# Authenticate
gcloud auth application-default login
```

Set your environment variables:
```bash
export GOOGLE_API_KEY="your-api-key"        # from aistudio.google.com/apikey
export GCP_PROJECT="your-gcp-project-id"
export GCP_BUCKET="your-gcs-bucket-name"
export GCP_REGION="us-central1"
```

---

## 📄 Dataset Format

Gemini uses a **different format from OpenAI**. Key differences:

| Field | OpenAI | Gemini |
|---|---|---|
| System prompt | `messages` with `role: system` | `systemInstruction` (top-level) |
| Messages key | `messages` | `contents` |
| AI role name | `assistant` | `model` |
| Content key | `content` | `parts: [{text: "..."}]` |

### Single-Turn Example
```jsonl
{
  "systemInstruction": {
    "role": "system",
    "parts": [{"text": "You are a helpful assistant for a junk food business."}]
  },
  "contents": [
    {"role": "user",  "parts": [{"text": "What items are on your menu?"}]},
    {"role": "model", "parts": [{"text": "We have Burgers, Fries, and Drinks! Combos from $8.99 🍔"}]}
  ]
}
```

### Multi-Turn Conversation Example
```jsonl
{
  "systemInstruction": {
    "role": "system",
    "parts": [{"text": "You are a helpful assistant for a junk food business."}]
  },
  "contents": [
    {"role": "user",  "parts": [{"text": "Do you have vegetarian options?"}]},
    {"role": "model", "parts": [{"text": "Yes! Veggie Burger, Onion Rings, and all Milkshakes are vegetarian!"}]},
    {"role": "user",  "parts": [{"text": "How much is the Veggie Burger?"}]},
    {"role": "model", "parts": [{"text": "Veggie Burger is $6.99 or $9.99 as a combo!"}]}
  ]
}
```

### Dataset Requirements
| Spec | Value |
|---|---|
| Format | `.jsonl` only |
| Recommended min examples | 100–500 |
| Max tokens per example | 131,072 |
| Max file size | 1 GB |
| Storage | Must be in **Google Cloud Storage (GCS)** |

---

## 📁 Project Structure

```
gemini-finetune/
│
├── data/
│   └── train.jsonl              # Your training dataset
│   └── validation.jsonl         # Optional validation set
│
├── scripts/
│   ├── prepare_dataset.py       # Generate & validate JSONL
│   ├── upload_to_gcs.py         # Upload dataset to GCS
│   ├── launch_finetune.py       # Start fine-tuning job
│   └── inference.py             # Use fine-tuned model
│
├── utils/
│   ├── view_contents.py         # Inspect dataset contents
│   ├── validate_dataset.py      # Validate JSONL format
│   └── count_tokens.py          # Count tokens per role
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```python
import json

# Step 1 — Prepare your dataset
data = [
    {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": "You are a domain expert assistant."}]
        },
        "contents": [
            {"role": "user",  "parts": [{"text": "Your question here"}]},
            {"role": "model", "parts": [{"text": "Your ideal answer here"}]}
        ]
    }
    # Add 100-500 more examples...
]

with open("data/train.jsonl", "w") as f:
    for entry in data:
        f.write(json.dumps(entry) + "\n")

print("✅ Dataset ready!")
```

---

## 🛠️ Helper Functions

### View Dataset Contents
```python
import json

def get_jsonl_contents(file_path: str):
    results = []
    with open(file_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for i, line in enumerate(lines, 1):
        example  = json.loads(line)
        sys_text = example.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
        results.append({
            "example_id": i,
            "system":     sys_text,
            "contents":   example.get("contents", [])
        })
    return results

# Usage
contents = get_jsonl_contents("data/train.jsonl")
print(contents[0]["contents"])   # → raw contents list
```

### Count Tokens Per Role
```python
def count_role_tokens(file_path: str):
    role_tokens = {"system": 0, "user": 0, "model": 0}
    with open(file_path, "r") as f:
        for line in f:
            example = json.loads(line.strip())
            sys_parts = example.get("systemInstruction", {}).get("parts", [])
            if sys_parts:
                role_tokens["system"] += len(sys_parts[0].get("text", "")) // 4
            for turn in example.get("contents", []):
                role = turn.get("role", "")
                text = turn.get("parts", [{}])[0].get("text", "")
                if role in role_tokens:
                    role_tokens[role] += len(text) // 4
    return role_tokens
```

### Evaluate Dataset Through Gemini
```python
import google.generativeai as genai

def evaluate_dataset_with_gemini(file_path: str, model_name: str = "gemini-2.5-flash"):
    genai.configure(api_key="YOUR_GOOGLE_API_KEY")
    results = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f, 1):
            example  = json.loads(line.strip())
            sys_text = example.get("systemInstruction", {}).get("parts", [{}])[0].get("text", "")
            contents = example.get("contents", [])
            user_msg = next((t["parts"][0]["text"] for t in reversed(contents) if t["role"] == "user"), "")
            expected = next((t["parts"][0]["text"] for t in reversed(contents) if t["role"] == "model"), "")
            model    = genai.GenerativeModel(model_name, system_instruction=sys_text)
            response = model.generate_content(user_msg)
            results.append({
                "example_id": i,
                "user_message": user_msg,
                "expected":    expected,
                "generated":   response.text   # ← model's response
            })
    return results
```

---

## 🚀 Launch Fine-Tuning Job

### Step 1 — Upload Dataset to GCS
```python
from google.cloud import storage

def upload_to_gcs(local_file, bucket_name, destination):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(destination).upload_from_filename(local_file)
    print(f"✅ Uploaded to gs://{bucket_name}/{destination}")

upload_to_gcs("data/train.jsonl", "your-bucket", "datasets/train.jsonl")
```

### Step 2 — Start Fine-Tuning Job (Vertex AI)
```python
import vertexai
from vertexai.tuning import sft

vertexai.init(project="your-project-id", location="us-central1")

tuning_job = sft.train(
    source_model="gemini-2.5-flash",
    train_dataset="gs://your-bucket/datasets/train.jsonl",
    validation_dataset="gs://your-bucket/datasets/validation.jsonl",  # optional
    epochs=3,
    tuned_model_display_name="my-domain-expert-model"
)

print(f"✅ Job started: {tuning_job.resource_name}")
```

### Step 3 — Monitor Job
```python
# Check status in Google Cloud Console
# https://console.cloud.google.com/vertex-ai/training/custom-jobs

tuning_job.refresh()
print(f"Status: {tuning_job.state}")
print(f"Model:  {tuning_job.tuned_model_endpoint_name}")
```

---

## 🎯 Use Your Fine-Tuned Model

```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="your-project-id", location="us-central1")

# Use your fine-tuned model endpoint
model = GenerativeModel("projects/your-project/locations/us-central1/endpoints/YOUR_ENDPOINT_ID")

response = model.generate_content("What items are on your menu?")
print(response.text)
```

---

## 🏗️ What Can You Build?

| Product | Description |
|---|---|
| 🤖 **Chatbot** | Domain-specific customer support bot |
| 🧠 **AI Agent** | Autonomous agent that takes real actions |
| 🏷️ **Classifier** | Auto-tag, route, and categorize inputs |
| 📄 **Extractor** | Parse structured data from unstructured text |
| 🔍 **Recommender** | Intent-based product/content recommender |
| ✍️ **Content Engine** | Generate brand-voice copy at scale |
| 📚 **Internal Tool** | Company knowledge base for staff |

---

## 📚 Resources

- [Vertex AI Supervised Fine-Tuning Docs](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Google AI Studio](https://aistudio.google.com)
- [Vertex AI Console](https://console.cloud.google.com/vertex-ai)

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

> Built with ❤️ | Fine-tune smarter, not harder 🚀
