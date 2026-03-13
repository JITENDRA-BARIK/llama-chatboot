import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()

BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME    = os.getenv("MODEL_NAME", "llama3.2")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Answer questions clearly and concisely."
)
HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "20"))

# ─── Build the LLM (Groq if API key set, else Ollama) ─────────────────────────
def build_llm():
    # Read fresh every request so Vercel env vars are always picked up
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        from langchain_groq import ChatGroq
        groq_model = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
        return ChatGroq(
            api_key=groq_key,
            model=groq_model,
            temperature=0.7,
        )
    else:
        chat_ollama_cls = None

        # Preferred package (newer)
        try:
            from langchain_ollama import ChatOllama as _ChatOllama
            chat_ollama_cls = _ChatOllama
        except ImportError:
            pass

        # Backward-compatible fallback
        if chat_ollama_cls is None:
            try:
                from langchain_community.chat_models import ChatOllama as _ChatOllama
                chat_ollama_cls = _ChatOllama
            except ImportError:
                pass

        if chat_ollama_cls is None:
            raise RuntimeError(
                "No GROQ_API_KEY set and no Ollama chat integration is installed. "
                "Install langchain-ollama or set GROQ_API_KEY."
            )

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        is_vercel = bool(os.environ.get("VERCEL"))

        if is_vercel and ("127.0.0.1" in base_url or "localhost" in base_url):
            raise RuntimeError(
                "This app is running on Vercel and cannot access local Ollama at "
                "127.0.0.1/localhost. Set GROQ_API_KEY in Vercel, or set "
                "OLLAMA_BASE_URL to a publicly reachable Ollama endpoint."
            )

        return chat_ollama_cls(
            model=os.environ.get("MODEL_NAME", "llama3.2"),
            base_url=base_url,
            temperature=0.7,
        )

# ─── Build the prompt template ────────────────────────────────────────────────
def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

# ─── Chatbot class ────────────────────────────────────────────────────────────
class Chatbot:
    def __init__(self):
        self.llm     = build_llm()
        self.prompt  = build_prompt()
        self.chain   = self.prompt | self.llm
        self.history: list = []

    def chat(self, user_input: str) -> str:
        response = self.chain.invoke({
            "history": self.history,
            "input":   user_input,
        })

        # Store conversation history
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response.content))

        # Trim history to avoid context overflow
        if len(self.history) > HISTORY_LIMIT * 2:
            self.history = self.history[-(HISTORY_LIMIT * 2):]

        return response.content

    def reset(self):
        self.history = []
        print("\n🔄  Conversation history cleared.\n")

# ─── CLI loop ─────────────────────────────────────────────────────────────────
def print_banner():
    print("=" * 60)
    print("  🦙  LLaMA Chatbot  |  Powered by LangChain + Ollama")
    print(f"       Model : {MODEL_NAME}")
    print("=" * 60)
    print("  Commands:")
    print("    'quit' or 'exit'  →  Exit the chatbot")
    print("    'reset' or 'clear' →  Clear conversation history")
    print("=" * 60)
    print()

def main():
    print_banner()

    bot = Chatbot()

    # Quick connectivity check
    print("⏳  Connecting to Ollama… ", end="", flush=True)
    try:
        bot.llm.invoke("Hi")
        print("✅  Connected!\n")
    except Exception as e:
        print(f"\n❌  Could not connect to Ollama at {BASE_URL}")
        print(f"    Error: {e}")
        print("\n💡  Make sure Ollama is running:  ollama serve")
        print(f"    And the model is pulled:       ollama pull {MODEL_NAME}\n")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("\n👋  Goodbye!")
            break

        if user_input.lower() in {"reset", "clear"}:
            bot.reset()
            continue

        print("Bot: ", end="", flush=True)
        try:
            reply = bot.chat(user_input)
            print(reply)
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
        print()


if __name__ == "__main__":
    main()
