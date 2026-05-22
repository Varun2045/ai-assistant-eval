import gradio as gr
import os
from huggingface_hub import InferenceClient

hf_token = os.getenv("HF_TOKEN", "")
client = InferenceClient(api_key=hf_token if hf_token else None)

def respond(message, history):
    messages = [{"role": "system", "content": "You are a helpful and polite open-source personal assistant."}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=messages,
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(
    respond,
    title="🦙 Qwen 2.5 Personal Assistant",
    description="This is your public Open Source Assistant hosted on Hugging Face Spaces."
)

if __name__ == "__main__":
    demo.launch()
