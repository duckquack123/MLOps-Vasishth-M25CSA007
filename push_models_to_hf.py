#!/usr/bin/env python
"""
Push Assignment 5 models to Hugging Face Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import login, upload_file, create_repo, HfApi

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USERNAME", "DuckyDuck123")

MODELS = [
    {
        "name": "Q1 Best LoRA (Optuna)",
        "repo_id": "vit-s-cifar100-lora-best",
        "local_path": "Q1/checkpoints/lora_r8_a4_d0.1_best",
        "type": "adapter",
        "description": "ViT-S with LoRA (Rank=8, Alpha=4) fine-tuned on CIFAR-100. Test Accuracy: 90.69%, Validation: 91.05%"
    },
    {
        "name": "Q2 ResNet-18 Clean Classifier",
        "repo_id": "resnet18-cifar10-clean",
        "local_path": "Q2/checkpoints/resnet18_cifar10_best.pt",
        "type": "model",
        "description": "ResNet-18 trained from scratch on CIFAR-10. Clean Accuracy: 89.86%"
    },
    {
        "name": "Q2 PGD Adversarial Detector",
        "repo_id": "resnet34-pgd-detector",
        "local_path": "Q2/checkpoints/detector_pgd_best.pt",
        "type": "detector",
        "description": "ResNet-34 binary detector for PGD adversarial images. Detection Accuracy: 98.30%"
    },
    {
        "name": "Q2 BIM Adversarial Detector",
        "repo_id": "resnet34-bim-detector",
        "local_path": "Q2/checkpoints/detector_bim_best.pt",
        "type": "detector",
        "description": "ResNet-34 binary detector for BIM adversarial images. Detection Accuracy: 86.02%"
    }
]

def main():
    if not HF_TOKEN:
        print("❌ HF_TOKEN not found in environment. Set it before running this script.")
        return False
    
    # Login
    print(f"🔐 Logging in to Hugging Face Hub with token...")
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("✅ Logged in successfully!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False
    
    api = HfApi()
    results = []
    
    for model in MODELS:
        repo_id = f"{HF_USER}/{model['repo_id']}"
        print(f"\n{'='*70}")
        print(f"📤 Pushing: {model['name']}")
        print(f"   Repo: {repo_id}")
        print(f"   Local: {model['local_path']}")
        print(f"{'='*70}")
        
        try:
            # Create repository if it doesn't exist
            try:
                api.repo_info(repo_id=repo_id)
                print(f"   ✅ Repository exists")
            except Exception:
                print(f"   📝 Creating new repository...")
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    exist_ok=True,
                    private=False
                )
                print(f"   ✅ Repository created")
            
            # Upload files
            if model["type"] == "adapter":
                # For adapter, upload all files in the directory
                local_dir = Path(model['local_path'])
                files_to_upload = []
                
                for file in local_dir.glob("**/*"):
                    if file.is_file():
                        files_to_upload.append(file)
                
                print(f"   📁 Found {len(files_to_upload)} files to upload:")
                for file in files_to_upload:
                    rel_path = file.relative_to(local_dir.parent)
                    print(f"      - {file.name}")
                    
                    url = upload_file(
                        path_or_fileobj=str(file),
                        path_in_repo=file.name,
                        repo_id=repo_id,
                        token=HF_TOKEN
                    )
                    print(f"        ✅ Uploaded: {file.name}")
            else:
                # For single .pt file
                local_file = Path(model['local_path'])
                if not local_file.exists():
                    print(f"   ❌ File not found: {model['local_path']}")
                    results.append(f"❌ {model['name']}: File not found")
                    continue
                
                # Get file size
                size_mb = local_file.stat().st_size / (1024**2)
                print(f"   📦 File size: {size_mb:.1f} MB")
                
                url = upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=local_file.name,
                    repo_id=repo_id,
                    token=HF_TOKEN
                )
                print(f"   ✅ Uploaded: {local_file.name}")
            
            # Success
            hf_link = f"https://huggingface.co/{repo_id}"
            print(f"   🎉 Model available at: {hf_link}")
            results.append(f"✅ {model['name']}: {hf_link}")
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            results.append(f"❌ {model['name']}: {str(e)}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 UPLOAD SUMMARY")
    print(f"{'='*70}")
    for result in results:
        print(result)
    
    return all("✅" in r for r in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
