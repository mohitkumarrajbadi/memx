"""Example: Using MemX as a LangChain-compatible memory wrapper.

This is a standalone stub — it does NOT require langchain to be installed.
It shows the pattern for integrating MemX into a LangChain agent.
"""

from memx import MemX


class MemXLangChainMemory:
    """Drop-in LangChain memory class backed by MemX.

    Usage with LangChain::

        from langchain.chains import ConversationChain
        from langchain.llms import OpenAI

        memory = MemXLangChainMemory()
        chain = ConversationChain(llm=OpenAI(), memory=memory)
        chain.run("Hello!")
    """

    memory_key: str = "history"

    def __init__(self, **kwargs):
        self.memx = MemX(**kwargs)

    # LangChain Memory interface

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict) -> dict:
        """Retrieve relevant context for the current input."""
        query = inputs.get("input", inputs.get("question", ""))
        if not query:
            return {self.memory_key: ""}
        results = self.memx.rag(str(query), top_k=5)
        context = "\n".join(f"- {r.content}" for r in results)
        return {self.memory_key: context}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Persist the conversation turn as a memory."""
        user_msg = inputs.get("input", inputs.get("question", ""))
        ai_msg = outputs.get("output", outputs.get("response", ""))
        if user_msg:
            self.memx.add(f"User said: {user_msg}")
        if ai_msg:
            self.memx.add(f"AI responded: {ai_msg}")

    def clear(self) -> None:
        self.memx.clear()


# ----- Demo (runs without langchain) -----
if __name__ == "__main__":
    mem = MemXLangChainMemory()

    # Simulate conversation turns
    mem.save_context(
        {"input": "I'm a rice farmer from Andhra Pradesh"},
        {"output": "Nice to meet you! I can help with farming queries."},
    )
    mem.save_context(
        {"input": "What's the best season to plant paddy?"},
        {"output": "Kharif season (June-October) is ideal for paddy cultivation."},
    )
    mem.save_context(
        {"input": "I decided to use SRI method for next season"},
        {"output": "Great choice! SRI can improve yields by 20-30%."},
    )

    # Retrieve context
    context = mem.load_memory_variables({"input": "What does the farmer grow?"})
    print("Retrieved context for 'What does the farmer grow?':")
    print(context["history"])
    print()

    print(f"Total memories: {mem.memx.stats()['total']}")
