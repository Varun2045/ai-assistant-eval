# -*- coding: utf-8 -*-
"""
Hugging Face Spaces Deployment Automation Script
Deploys a personal assistant running Qwen 2.5 on Hugging Face Spaces.
"""
import os
import sys
import shutil

def write_space_files(space_dir):
    os.makedirs(space_dir, exist_ok=True)
    
    # 1. Write the Streamlit app.py for the Hugging Face Space
    app_code = """import streamlit as st
import os
from huggingface_hub import InferenceClient

# Page Config
st.set_page_config(page_title="OSS Personal Assistant (Qwen)", page_icon="🦙", layout="centered")

st.title("🦙 Qwen 2.5 Personal Assistant")
st.markdown("This is your public Open Source Assistant hosted on Hugging Face Spaces.")

# Initialize client (uses Hugging Face serverless API)
hf_token = os.getenv("HF_TOKEN", "")
client = InferenceClient(api_key=hf_token if hf_token else None)

# Initialize Session State Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask Qwen something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Qwen is thinking..."):
            try:
                # Format system context and conversation history
                messages = [{"role": "system", "content": "You are a helpful and polite open-source personal assistant."}]
                for m in st.session_state.messages:
                    messages.append({"role": m["role"], "content": m["content"]})
                
                # Query serverless inference API
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error calling Qwen API: {e}")
"""
    with open(os.path.join(space_dir, "app.py"), "w", encoding="utf-8") as f:
        f.write(app_code)

    # 2. Write requirements.txt
    requirements = """streamlit
huggingface_hub
"""
    with open(os.path.join(space_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(requirements)

    # 3. Write README.md metadata for HF Spaces
    readme = """---
title: Qwen Personal Assistant
emoji: 🦙
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Qwen 2.5 Personal Assistant
A clean, conversational personal assistant running on Hugging Face Serverless Inference.
"""
    with open(os.path.join(space_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"Hugging Face Space source code created in directory: {space_dir}")

def deploy_space(space_dir, repo_id, token):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub library is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"Creating Space repository: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit", exist_ok=True)
        print("Uploading files to Space...")
        api.upload_folder(
            folder_path=space_dir,
            repo_id=repo_id,
            repo_type="space"
        )
        # Upload the Hugging Face token as a Space secret for Serverless Inference (optional but ensures quota limit is high)
        try:
            api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
            print("Successfully configured HF_TOKEN secret inside Space for high API limit.")
        except Exception as se:
            print(f"Warning: Could not set HF_TOKEN space secret: {se}")
            
        print(f"\\n🚀 Public OSS Assistant successfully deployed! View it here: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    space_dir = "./hf_space"
    write_space_files(space_dir)
    
    print("\n--- Hugging Face Spaces Deployment ---")
    print("This script will create a public Space repository and upload the Streamlit helper.")
    
    if len(sys.argv) > 2:
        repo_id = sys.argv[1]
        token = sys.argv[2]
        deploy_space(space_dir, repo_id, token)
    else:
        print("\nTo deploy, run the script with your Hugging Face Space Repo ID and token:")
        print("  python deploy_hf.py <username/space-name> <hf-write-token>")
        print("\nExample:")
        print("  python deploy_hf.py varun/qwen-assistant hf_xxxxxxxxxxxxxxxxxxxxxxxxxx")
