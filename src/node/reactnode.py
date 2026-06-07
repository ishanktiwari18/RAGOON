"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional

from src.state.rag_state import RAGState

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from local vector store."""
        docs = self.retriever.invoke(state.question)

        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self) -> List:
        """Build retriever + Wikipedia tools."""

        retriever_func = self.retriever

        @tool
        def search_documents(query: str):
            """Search the local document corpus for relevant information."""
            docs = retriever_func.invoke(query)

            if not docs:
                return "No documents found."

            merged = []

            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                content = d.page_content[:1500]

                merged.append(f"[{i}] {title}\n{content}")

            return "\n\n".join(merged)

        wiki_api = WikipediaAPIWrapper(top_k_results=3, lang="en")

        @tool
        def search_wikipedia(query: str):
            """Search Wikipedia for general knowledge and wide-ranging facts."""
            try:
                return wiki_api.run(query)
            except Exception as e:
                return f"Wikipedia search failed temporarily: {str(e)}"

        return [search_documents, search_wikipedia]

    def _build_agent(self):
        """Build ReAct agent with tools."""

        tools = self._build_tools()

        system_prompt = (
            "You are a helpful RAG agent. "
            "First use search_documents for questions related to the loaded research papers. "
            "Use search_wikipedia only when broader general knowledge is needed. "
            "If a tool fails, continue using available context. "
            "Return only the final useful answer."
        )

        self._agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=system_prompt
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using ReAct agent with retriever + Wikipedia."""

        if self._agent is None:
            self._build_agent()

        try:
            result = self._agent.invoke(
                {"messages": [HumanMessage(content=state.question)]}
            )

            messages = result.get("messages", [])
            answer: Optional[str] = None

            if messages:
                answer_msg = messages[-1]
                answer = getattr(answer_msg, "content", None)

            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=answer or "Could not generate answer."
            )

        except Exception as e:
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=f"Agent error: {str(e)}"
            )