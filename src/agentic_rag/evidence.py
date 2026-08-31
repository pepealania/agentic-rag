from __future__ import annotations

from agentic_rag.retrieval import VectorStore
from agentic_rag.state import AgentState


class EvidenceAgent:
    """Evidence agent with a bounded retrieval/reformulation capability."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def search(self, query: str):
        """Search the vector store using the supplied query."""

        return self.vector_store.search(
            query,
            top_k=self.top_k,
        )

    def reformulate(self, query: str) -> str:
        """Create one deterministic reformulation of a query."""

        return f"{query} relevant information"

    def run(self, state: AgentState) -> AgentState:
        """Retrieve evidence, allowing at most one reformulation."""

        question = state["question"]
        query = question.question

        results = self.search(query)

        tool_results = list(state.get("tool_results", []))
        tool_results.append(
            {
                "tool": "vector_search",
                "query": query,
                "result_count": len(results),
            }
        )

        query_used = query

        if not results:
            reformulated = self.reformulate(query)

            tool_results.append(
                {
                    "tool": "query_reformulation",
                    "original_query": query,
                    "reformulated_query": reformulated,
                }
            )

            results = self.search(reformulated)
            query_used = reformulated

            tool_results.append(
                {
                    "tool": "vector_search",
                    "query": reformulated,
                    "result_count": len(results),
                }
            )

        return {
            "query_used": query_used,
            "retrieved_chunks": results,
            "tool_results": tool_results,
        }
