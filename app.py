import os
from flask import Flask, render_template, request, jsonify, session
from chatbot import Chatbot

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Store chatbot instances per session
chatbots = {}

def get_bot():
    sid = session.get("sid")
    if sid not in chatbots:
        chatbots[sid] = Chatbot()
    return chatbots[sid]

@app.before_request
def ensure_session():
    if "sid" not in session:
        import uuid
        session["sid"] = str(uuid.uuid4())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400
    try:
        bot = get_bot()
        reply = bot.chat(user_input)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    bot = get_bot()
    bot.reset()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
