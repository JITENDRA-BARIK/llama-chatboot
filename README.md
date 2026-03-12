# 🦙 LLaMA Chatbot — LangChain + Ollama

A simple conversational chatbot powered by **LLaMA** (via [Ollama](https://ollama.com)) and **LangChain**.

---

## 📁 Project Structure

```
Project 2/
├── chatbot.py        # Main chatbot script
├── requirements.txt  # Python dependencies
├── .env              # Configuration (model, prompt, etc.)
└── README.md
```

---

## ⚙️ Prerequisites

### 1. Install Ollama
Download and install Ollama from [https://ollama.com/download](https://ollama.com/download).

### 2. Pull the LLaMA model
```bash
ollama pull llama3.2
```
> You can also use `llama3`, `llama3.1`, `mistral`, etc. — just update `MODEL_NAME` in `.env`.

### 3. Start Ollama server
```bash
ollama serve
```
> Ollama usually starts automatically after installation.

---

## 🚀 Setup & Run

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Edit `.env`
```env
MODEL_NAME=llama3.2                  # Change to any Ollama model
SYSTEM_PROMPT=You are a helpful AI.  # Customize the bot's personality
CHAT_HISTORY_LIMIT=20                # Max conversation turns kept in memory
```

### 3. Run the chatbot
```bash
python chatbot.py
```

---

## 💬 Usage

```
============================================================
  🦙  LLaMA Chatbot  |  Powered by LangChain + Ollama
       Model : llama3.2
============================================================
  Commands:
    'quit' or 'exit'   →  Exit the chatbot
    'reset' or 'clear' →  Clear conversation history
============================================================

You: Hello!
Bot: Hi there! How can I help you today?

You: reset
🔄  Conversation history cleared.

You: exit
👋  Goodbye!
```

---

## 🔧 Configuration (`.env`)

| Variable            | Default                          | Description                        |
|---------------------|----------------------------------|------------------------------------|
| `OLLAMA_BASE_URL`   | `http://localhost:11434`         | Ollama server URL                  |
| `MODEL_NAME`        | `llama3.2`                       | LLaMA model to use                 |
| `SYSTEM_PROMPT`     | *helpful assistant*              | Bot personality / instructions     |
| `CHAT_HISTORY_LIMIT`| `20`                             | Max turns stored in memory         |
