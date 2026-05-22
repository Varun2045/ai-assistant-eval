import streamlit as st
import os
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Qwen 2.5 Personal Assistant", page_icon="🦙")
st.title("🦙 Qwen 2.5 Personal Assistant")
st.write("This is your public Open Source Assistant hosted on Hugging Face Spaces.")

hf_token = os.getenv("HF_TOKEN", "")
client = InferenceClient(api_key=hf_token if hf_token else None)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Build history for API
        api_messages = [{"role": "system", "content": "You are a helpful and polite open-source personal assistant."}]
        for msg in st.session_state.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
            
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=api_messages,
                max_tokens=512,
                temperature=0.7
            )
            full_response = response.choices[0].message.content
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            error_msg = f"⚠️ API Error: {str(e)}"
            if "503" in str(e) or "loading" in str(e).lower():
                error_msg = "⏳ The Qwen model is currently waking up from sleep to save server costs. Please wait about 30 seconds and send your message again!"
            message_placeholder.markdown(error_msg)
