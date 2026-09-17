# Gen-AI-Basics

# 🧠 Gen AI Basics

## 📑 Table of Contents
 
- [What is an LLM?](#-what-is-an-llm)
- [Model vs LLM](#-model-vs-llm)
- [Popular LLMs](#-popular-llms)
- [Using LLMs via SDKs](#-using-llms-via-sdks)
- [The Multi-SDK Problem](#-the-problem)
- [LangChain — The Solution](#-langchain--the-solution)
- [LangChain Core Components](#-langchain-core-components)
  - [1. Models](#1️⃣-models)
  - [2. Prompts](#2️⃣-prompts)
  - [3. Chains](#3️⃣-chains)
  - [4. Memory](#4️⃣-memory)
  - [5. Indexes](#5️⃣-indexes)
  - [6. Agents](#6️⃣-agents)
- [Quick Revision Checklist](#-quick-revision-checklist)
---
 
## ❓ What is an LLM?
 
> **LLM = Large Language Model**
 
Break the term into 3 words:
 
| Word | Meaning |
|---|---|
| **Large** | Trained on a *huge* amount of text data (books, websites, Wikipedia, articles, code, conversations, etc.) |
| **Language** | Understands & generates human language — English, Hindi, Spanish... even programming languages like Python |
| **Model** | A deep learning system (a neural network) that learns patterns from data |
 
```mermaid
flowchart LR
    A["User types:\n'What is Python?'"] --> B{LLM}
    B -->|"NOT this"| C["❌ Searches Google\n❌ Thinks like a human\n❌ Understands meaning like we do"]
    B -->|"Actually does this"| D["✅ Looks at learned patterns"]
    D --> E["✅ Predicts next likely word"]
    E --> F["✅ Keeps predicting word by word"]
    F --> G["Final Answer"]
```
 
> 💡 **Analogy:** It's like your phone's keyboard autocomplete — but supercharged with billions of parameters and trained on massive datasets.
 
---
 
## 🧠 Model vs LLM
 
```mermaid
flowchart TD
    A["Model (umbrella term)"] --> B["Prediction Models\n(house price, spam detection)"]
    A --> C["Embedding Models\n(text → vectors)"]
    A --> D["LLMs\n(GPT, Claude, Gemini)"]
    A --> E["Multimodal Models\n(text + image + audio)"]
```
 
- **Model** = any trained AI system that learns patterns from data.
- **LLM** = a *specific type* of model that specializes in understanding and generating **language**.
- Every LLM is a model, but not every model is an LLM (e.g. an embedding model isn't an LLM).
---
 
## ⭐ Popular LLMs
 
| Model Family | Provider |
|---|---|
| GPT Models | OpenAI |
| Gemini Models | Google DeepMind |
| LLaMA | Meta |
| Claude | Anthropic |
| Grok | xAI |
| Mistral | Mistral AI |
 
> 📌 In Generative AI, we **don't build these models from scratch** — we learn how to use these **pre-built LLMs** effectively.
 
---
 
## 💻 Using LLMs via SDKs
 
<details>
<summary><b>OpenAI</b></summary>
```python
from openai import OpenAI
 
client = OpenAI()
 
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what is a Large Language Model in simple terms."}
    ],
)
 
print(response.choices[0].message.content)
```
</details>
<details>
<summary><b>Gemini</b></summary>
```python
import google.generativeai as genai
import os
 
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
 
model = genai.GenerativeModel("gemini-1.5-flash")
 
response = model.generate_content(
    "Explain what is a Large Language Model in simple terms."
)
 
print(response.text)
```
</details>
<details>
<summary><b>Claude</b></summary>
```python
import anthropic
import os
 
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
 
response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "Explain what is a Large Language Model in simple terms."}
    ],
)
 
print(response.content[0].text)
```
</details>
---
 
## ⚠️ The Problem
 
```mermaid
flowchart LR
    A[Your AI App] -->|different syntax| B[OpenAI SDK]
    A -->|different syntax| C[Gemini SDK]
    A -->|different syntax| D[Claude SDK]
    B --> E["😵 Hard to scale /\nswap providers"]
    C --> E
    D --> E
```
 
If every provider has a **different SDK**, **different syntax**, and a **different response format** — how do we build scalable AI applications?
 
---
 
## 🔗 LangChain — The Solution
 
```mermaid
flowchart LR
    A[Your AI App] --> B[LangChain]
    B --> C[OpenAI]
    B --> D[Gemini]
    B --> E[Claude]
```
 
**LangChain** is a framework that standardizes how you talk to different LLM providers — write your logic once, swap models/providers without rewriting everything.
 
---
 
## 🧩 LangChain Core Components
 
```mermaid
flowchart TD
    LC[LangChain] --> M[Models]
    LC --> P[Prompts]
    LC --> C[Chains]
    LC --> Mem[Memory]
    LC --> I[Indexes]
    LC --> A[Agents]
```
 
### 1️⃣ Models
 
The **core intelligence layer** — connects your app to LLM providers (OpenAI, Google DeepMind, Anthropic).
 
```mermaid
flowchart LR
    Text["Your input text"] --> Chat["Chat / Language Model\n(answer, write code, summarize)"]
    Text --> Embed["Embedding Model\n(text → vector numbers)"]
    Text --> Multi["Multimodal Model\n(text + image + audio)"]
```
 
| Type | What it does |
|---|---|
| **Chat / Language Models** | Generate text, answer questions, write code, summarize content |
| **Embedding Models** | Convert text → numbers (vectors); used in search & RAG |
| **Multimodal Models** | Work with more than text — images, audio, files |
 
---
 
### 2️⃣ Prompts
 
A **prompt** is the instruction given to an LLM — it tells the model *what to do*, *how to respond*, and *what format to use*.
 
> 📌 Better prompt → Better output. LLMs aren't mind readers.
 
```mermaid
flowchart TD
    P[Prompt Types] --> S["Simple Prompt\n'Explain ML.'"]
    P --> SU["System + User Prompt\nSystem: role → User: question"]
    P --> T["Prompt Template\n'Explain {topic} in simple terms.'"]
    P --> O["Structured Output\n'Respond in JSON: {...}'"]
```
 
<details>
<summary><b>Examples for each type</b></summary>
**Simple Prompt**
```
Explain what is Machine Learning.
```
 
**System + User Prompt**
```
System: You are a helpful AI teacher.
User: Explain LLMs in simple terms.
```
 
**Prompt Template**
```
Explain {topic} in simple terms.
```
 
**Structured Output**
```
Respond in JSON format with keys: definition, example.
```
</details>
---
 
### 3️⃣ Chains
 
A **Chain** is a sequence of connected steps.
 
```mermaid
flowchart LR
    A["Input:\n'Summarize this article\nand translate to Hindi'"] --> B[Summarize text]
    B --> C[Translate summary]
    C --> D[Return final result]
```
 
Instead of `One Prompt → One Response`, a chain does `Step 1 → Step 2 → Step 3 → Final Output`. Real AI applications are rarely single-step.
 
---
 
### 4️⃣ Memory
 
Allows the AI system to **remember past interactions**.
 
```mermaid
flowchart TD
    U1["User: My name is Akarsh."] --> M{Memory enabled?}
    M -->|No| N1["User: What is my name?"] --> R1["❌ Model doesn't know"]
    M -->|Yes| N2["User: What is my name?"] --> R2["✅ Model: Akarsh"]
```
 
---
 
### 5️⃣ Indexes
 
LLMs only know what they were trained on. **Indexes** connect external data to an LLM.
 
```mermaid
flowchart LR
    LLM["LLM (trained knowledge only)"] -->|"❌ can't access"| D1[Company data]
    LLM -->|"❌ can't access"| D2[Private documents]
    LLM -->|"❌ can't access"| D3[Real-time updates]
    D1 & D2 & D3 --> R["Retrieval\n(Index / RAG)"] --> LLM2["LLM + External Knowledge"]
```
 
**Why needed?** LLMs don't know your company data, private documents, or real-time info.
**Solution:** Connect external knowledge → this system is called **Retrieval** (the "R" in **RAG**).
 
---
 
### 6️⃣ Agents
 
AI systems that **decide what action to take** — dynamically.
 
```mermaid
flowchart TD
    U[User query] --> AG{Agent}
    AG -->|needs web info| T1[🔍 Web Search Tool]
    AG -->|needs math| T2[🧮 Calculator Tool]
    AG -->|needs live data| T3[🔌 API Call]
    AG -->|needs records| T4[🗄️ Database Tool]
    T1 & T2 & T3 & T4 --> Res[Final Response]
```
 
| Chains | Agents |
|---|---|
| Follow **fixed** steps | **Decide** the next step dynamically |
 
A normal LLM flow is `User → Prompt → Response`. But if a task needs web search, calculations, API calls, or a database — the LLM alone can't do that. It needs **tools**, and an **Agent** decides which tool to use and when.
 
---
 
## 📝 Quick Revision Checklist
 
- [ ] Can explain LLM in 3 words (Large, Language, Model)
- [ ] Can explain Model vs LLM difference
- [ ] Can name 4+ popular LLMs and their providers
- [ ] Understand why raw provider SDKs don't scale
- [ ] Can explain all 6 LangChain components in one line each
- [ ] Know all 4 types of prompts with an example each
- [ ] Understand Chains vs Agents (fixed steps vs dynamic decisions)
- [ ] Understand Memory (context retention) with an example
- [ ] Understand Indexes/Retrieval and how it connects to RAG
 
