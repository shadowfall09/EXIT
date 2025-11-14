#!/usr/bin/env python3
"""
Upload EXIT LoRA adapter to Hugging Face Hub
"""
from huggingface_hub import HfApi, create_repo
import os

# Configuration
MODEL_PATH = "/mnt/data2/yichengtao/EXIT/outputs/exit_model/final_model"
REPO_ID = "Yugong09/exit_reproduction"

print(f"Uploading EXIT LoRA adapter...")
print(f"From: {MODEL_PATH}")
print(f"To: {REPO_ID}")
print()

# Check files exist
required_files = ["adapter_config.json", "adapter_model.safetensors", "README.md"]
for file in required_files:
    file_path = os.path.join(MODEL_PATH, file)
    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file}")
        exit(1)
    else:
        print(f"✓ Found: {file}")

print()

# Initialize API
api = HfApi()

# Create repo if needed
try:
    create_repo(REPO_ID, exist_ok=True, repo_type="model")
    print(f"✓ Repository ready: {REPO_ID}")
except Exception as e:
    print(f"Note: {e}")

print()
print("Uploading files...")

# Upload each file
for file in required_files:
    file_path = os.path.join(MODEL_PATH, file)
    print(f"  Uploading {file}...", end=" ")
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=REPO_ID,
            commit_message=f"Upload {file}"
        )
        print("✓")
    except Exception as e:
        print(f"✗ Failed: {e}")

print()
print(f"✅ Upload complete!")
print(f"🔗 View at: https://huggingface.co/{REPO_ID}")
