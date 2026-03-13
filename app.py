import os
from flask import Flask, render_template, request, jsonify
from chatbot import HISTORY_LIMIT, build_llm, build_prompt
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-me-in-production")


def _runtime_provider() -> str:
    return "groq" if os.environ.get("GROQ_API_KEY", "").strip() else "ollama"


def _friendly_error_message(error: Exception) -> str:
    raw = str(error)
    text = raw.lower()

    if "groq_api_key" in text or "api key" in text and "groq" in text:
        return "GROQ key is missing. Set GROQ_API_KEY in environment variables, or use Ollama locally."

    if "model_decommissioned" in text or "decommissioned" in text and "groq" in text:
        return (
            "Your configured GROQ_MODEL is deprecated. Set GROQ_MODEL to a supported model "
            "(for example: llama-3.1-8b-instant) and redeploy."
        )

    if any(token in text for token in [
        "localhost:11434",
        "127.0.0.1:11434",
        "connection refused",
        "failed to establish a new connection",
        "connecterror",
        "timed out",
        "ollama",
    ]):
        if os.getenv("VERCEL"):
            return (
                "This deployment is on Vercel, so local Ollama (127.0.0.1/localhost) "
                "is not reachable. Set GROQ_API_KEY in Vercel Environment Variables and "
                "redeploy, or set OLLAMA_BASE_URL to a public Ollama endpoint."
            )
        return (
            "Cannot reach Ollama server. Start Ollama (ollama serve) and ensure "
            "OLLAMA_BASE_URL is correct (default: http://127.0.0.1:11434)."
        )

    return raw

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "provider": _runtime_provider(),
        "groq_key_present": bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "groq_model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "vercel": bool(os.environ.get("VERCEL")),
        "vercel_env": os.environ.get("VERCEL_ENV", ""),
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "").strip()
    history_raw = data.get("history", [])  # [{"role":"user"|"bot", "text":"..."}]

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    try:
        # Rebuild LangChain history from client-sent history
        history = []
        for msg in history_raw[-(HISTORY_LIMIT * 2):]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["text"]))
            else:
                history.append(AIMessage(content=msg["text"]))

        llm    = build_llm()
        prompt = build_prompt()
        chain  = prompt | llm

        response = chain.invoke({"history": history, "input": user_input})
        return jsonify({"reply": response.content})

    except Exception as e:
        return jsonify({"error": _friendly_error_message(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
