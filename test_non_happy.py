from pathlib import Path

from agentic_rag.ingestion import load_documents
from agentic_rag.chunking import chunk_documents
from agentic_rag.embeddings import EmbeddingModel
from agentic_rag.retrieval import VectorStore
from agentic_rag.evidence import EvidenceAgent
from agentic_rag.analyst import AnalystAgent
from agentic_rag.validation import AnswerValidator
from agentic_rag.workflow import AgenticRAGWorkflow
from agentic_rag.schemas import Question


class FailOnceValidator(AnswerValidator):
    """Test validator that fails once, then uses the real validator."""

    def __init__(self):
        self.failed_once = False

    def run(self, state):
        if not self.failed_once:
            self.failed_once = True

            decision_log = list(state.get("decision_log", []))
            decision_log.append(
                {
                    "agent": "validator",
                    "decision": "forced_test_failure",
                    "error_count": 1,
                }
            )

            return {
                "validation_errors": ["forced_test_failure"],
                "decision_log": decision_log,
                "step_count": state["step_count"],
            }

        return super().run(state)


docs = load_documents(Path("data/raw"))
chunks = chunk_documents(docs, 500, 50)

model = EmbeddingModel(
    "sentence-transformers/all-MiniLM-L6-v2"
)

store = VectorStore(model)
store.build(chunks)

evidence = EvidenceAgent(
    store,
    top_k=5,
)

validator = FailOnceValidator()

workflow = AgenticRAGWorkflow(
    evidence_agent=evidence,
    analyst_agent=AnalystAgent(),
    validator=validator,
    max_steps=8,
)

question = Question(
    question_id="q01",
    question="What is Agentic RAG?",
)

result = workflow.run(
    {
        "question": question,
    }
)

print()
print("=" * 60)
print("NON-HAPPY PATH TEST")
print("=" * 60)

print()
print("--- RESULT ---")
print("route=", result.get("route"))
print("steps=", result.get("step_count"))
print("retries=", result.get("retry_count"))
print("validation_errors=", result.get("validation_errors"))

print()
print("--- DECISIONS ---")
for item in result.get("decision_log", []):
    print(item)

print()
print("--- TOOLS ---")
for item in result.get("tool_results", []):
    print(item)

print()
print("--- FINAL ANSWER ---")
print(result.get("answer"))
