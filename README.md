<h3 align="center">🤖 AI Assistant Arena & Evaluator</h3>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<div align="center">

**A premium, observability-enabled evaluation dashboard and interactive playground to compare Open Source (OSS) and Frontier LLMs across safety, factual accuracy, and hallucination rate paradigms.**

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">🌐 Live Demo</h3>

<div align="center">

**[📊 Try the Main Evaluation Dashboard Here!](https://ai-assistant-eval-g7wrhuzeaom8p5zunuxgpd.streamlit.app)**  
*(Hosted on Streamlit Community Cloud)*

<br>

**[🦙 Try the Qwen OSS Assistant Here!](#)**  
*(Replace the link above with your Hugging Face Space URL after running deploy_hf.py)*

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">🚀 What It Does</h3>

<div align="center">

*"Compare Open Source vs Frontier models side-by-side"*

*"Evaluate outputs with an impartial LLM-as-a-Judge"*

*"Monitor and block unsafe requests with Dual-Layer Guardrails"*

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">🛠️ Tech Stack</h3>

<div align="center">

Python 3.10+
<br>Streamlit (Interactive Dashboard)
<br>Groq API (Llama 3.3 70B & 3.1 8B Classifier)
<br>Gemini API (Gemini 2.5 Flash)
<br>SQLite (Observability Logging)
<br>Plotly (Visualizations)

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">⚙️ Setup</h3>

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/ai-assistant-eval
cd ai-assistant-eval
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY="your-groq-api-key-here"
GEMINI_API_KEY="your-gemini-api-key-here"
```

**4. Run the Application**
```bash
streamlit run app.py
```

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">🔧 Core Features</h3>

<div align="center">

| Feature | Description |
|---------|-------------|
| **Interactive Chat Arena** | Parallel multi-turn messaging panel for OSS and Frontier assistants. |
| **Dual-Layer Guardrails** | Uses `llama-3.1-8b-instant` to block toxic/jailbreak inputs and outputs. |
| **LLM-as-a-Judge Evaluator** | Impartial grading (1-10) with reasoning by `Llama 3.3 70B`. |
| **System Observability** | Real-time logging of prompts, guardrails, and latency via local SQLite. |
| **Benchmark Manager** | Create, edit, export/import prompt evaluation suites in JSON/CSV. |

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>



<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">⚖️ Architectural Analysis & Trade-Offs</h3>

<div align="center">

| Component | Choice | Trade-Off |
|-----------|--------|-----------|
| **Guardrails** | `llama-3.1-8b-instant` with zero-shot classification | Extra 150-250ms latency vs. local embedding/regex, but allows complex rule evaluations. |
| **Judge Model** | `Llama 3.3 70B` (Groq) | Near-frontier level reasoning with higher rate limits and speed than free-tier API models. |
| **Logging** | Local SQLite (`observability.db`) | Zero-config and zero-cost local setup, but limits horizontal scalability compared to external observability platforms. |

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">💸 OSS Deployment Cost & Latency</h3>

<div align="center">

| OSS Model | Deployment Target | Est. Latency | Est. Cost |
|-----------|-------------------|--------------|-----------|
| **Qwen2.5-0.5B-Instruct** | Hugging Face Spaces (Serverless API) | ~0.8s - 1.2s / request | $0.00 (Free Tier) |
| **Llama-3.1-8B** (Alternative) | Groq API (Cloud) | ~0.3s - 0.5s / request | Free Tier ($0 for eval) |
| **Qwen2.5-7B-Instruct** (Alternative) | Hugging Face Dedicated Endpoint (T4 GPU) | ~1.5s - 2.5s / request | ~$0.60 / hour |

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">🚀 Future Improvements</h3>

<div align="center">

**1. Guardrail Quantization:** Port custom safety classification models directly into local ONNX runtime instances inside the container to eliminate the API network hop completely.<br><br>
**2. Multi-Judge Consensus:** Implement a voting consensus protocol (e.g. Llama-70B, GPT-4o-Mini, and Gemini-Flash) to average out judge bias and handle outlier scores.<br><br>
**3. Conversational Drift Observability:** Log embedding vectors of conversation history to detect user topic drift and predict potential jailbreak sequences before they trigger a violation.

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<h3 align="center">📸 Demo</h3>

<div align="center">

<h4>🥊 Interactive Chat Arena</h4>

<br>

<img src="assets/Interactive%20Chat%20Arena.png" alt="Chat Arena View" width="800"/>

<br><br>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<br>

<h4>🛡️ Dual-Layer Safety Guardrails</h4>

<br>

<img src="assets/Guardrail%20Interception.png" alt="Safety Guardrail Alert" width="800"/>

<br><br>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<br>

<h4>📊 LLM-as-a-Judge Evaluation Dashboard</h4>

<br>

<img src="assets/LLM%20as%20a%20Judge%20Evaluation%20Dashboard.png" alt="Evaluation Dashboard" width="800"/>

<br><br>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>

<br>

<h4>📈 Real-Time System Observability Logs</h4>

<br>

<img src="assets/Real-Time%20System%20Observability%20Logs.png" alt="System Observability" width="800"/>

<br><br>

</div>

<div align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"></div>
