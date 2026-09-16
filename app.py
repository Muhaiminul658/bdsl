import os
import json
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import HfApi, hf_hub_download

app = FastAPI(title="Sign Trainer Backend")

# ---- CORS: সব origin allow (production-এ নির্দিষ্ট ডোমেইন দিন) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # production-এ: ["https://yourname.github.io"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Configuration (Render Environment Variables-এ দিন) ----
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_REPO_ID   = os.environ.get("HF_DATASET_REPO", "")
DATA_FILENAME = "sign_dataset.json"

api = HfApi(token=HF_TOKEN or None)


# ---- Pydantic ----
class SaveRequest(BaseModel):
    dataset: dict


# ---- HF Dataset Helpers ----
def download_dataset() -> dict:
    if not HF_TOKEN or not HF_REPO_ID:
        return {}
    try:
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=DATA_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def upload_dataset(data: dict):
    if not HF_TOKEN or not HF_REPO_ID:
        raise HTTPException(status_code=500, detail="HF_TOKEN বা HF_DATASET_REPO সেট করা নেই")

    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / DATA_FILENAME
        local_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=DATA_FILENAME,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message="Update sign dataset",
        )


# ---- Endpoints ----
@app.get("/")
def root():
    return {"status": "ok", "service": "sign-trainer-backend"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/dataset")
def get_dataset():
    return {"dataset": download_dataset()}


@app.post("/dataset")
def save_dataset(req: SaveRequest):
    existing = download_dataset()
    existing.update(req.dataset)   # merge: একই label থাকলে replace
    upload_dataset(existing)
    return {"status": "saved", "labels": list(existing.keys())}


@app.delete("/dataset/{label}")
def delete_label(label: str):
    existing = download_dataset()
    if label not in existing:
        raise HTTPException(status_code=404, detail=f"Label '{label}' পাওয়া যায়নি")
    del existing[label]
    upload_dataset(existing)
    return {"status": "deleted", "label": label}
