import os
from flask import Flask, render_template, request, jsonify
from chatbot import Chatbot, HISTORY_LIMIT, SYSTEM_PROMPT, build_llm, build_prompt
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-me-in-production")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
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
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
