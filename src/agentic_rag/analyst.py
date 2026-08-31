from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agentic_rag.schemas import Answer, Citation
from agentic_rag.state import AgentState


class AnalystAgent:
    """LLM-backed analyst with bounded, structured output."""

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = client or OpenAI(
            base_url=base_url,
            api_key="ollama",
        )

    def _build_prompt(
        self,
        question: str,
        retrieved_chunks,
    ) -> str:
        evidence = []

        for chunk in retrieved_chunks:
            evidence.append(
                f"""
DOCUMENT_ID: {chunk.document_id}
CHUNK_ID: {chunk.chunk_id}
SOURCE: {chunk.source}
SCORE: {chunk.score}

TEXT:
{chunk.text}
"""
            )

        evidence_text = "\n".join(evidence)

        return f"""
Eres el agente analista de un sistema RAG.

Debes responder EXCLUSIVAMENTE usando la evidencia proporcionada.
No uses conocimiento externo.

Pregunta:
{question}

Evidencia:
{evidence_text}

Reglas:
1. Cada hecho debe estar respaldado por la evidencia.
2. Los indicadores deben ser datos, métricas o señales explícitamente presentes
   en la evidencia.
3. Si la evidencia es insuficiente, indícalo en uncertainty y establece
   abstained=true.
4. No inventes citas.
5. human_review=true si existe incertidumbre relevante, evidencia insuficiente
   o una conclusión que requiera revisión humana.
6. Las citas deben usar exactamente los document_id, chunk_id y source
   proporcionados por la evidencia.

Devuelve exclusivamente JSON válido con esta estructura:

{{
  "answer": "respuesta breve en español",
  "facts": ["hecho respaldado por evidencia"],
  "indicators": ["indicador respaldado por evidencia"],
  "uncertainty": ["incertidumbre o limitación"],
  "citations": [
    {{
      "document_id": "DOC-ID",
      "chunk_id": "chunk-ID",
      "source": "SOURCE"
    }}
  ],
  "human_review": false,
  "abstained": false
}}
"""

    def _parse_response(
        self,
        raw_response: str,
        question_id: str,
    ) -> Answer:
        raw = raw_response.strip()

        if raw.startswith("```"):
            raw = raw.replace("```json", "", 1)
            raw = raw.replace("```", "")
            raw = raw.strip()

        data = json.loads(raw)

        data["question_id"] = question_id

        return Answer.model_validate(data)

    def run(self, state: AgentState) -> AgentState:
        question = state["question"]
        retrieved_chunks = state.get("retrieved_chunks", [])

        decision_log = list(state.get("decision_log", []))

        if not retrieved_chunks:
            answer = Answer(
                question_id=question.question_id,
                answer=(
                    "No se encontró evidencia suficiente para responder "
                    "la pregunta."
                ),
                facts=[],
                indicators=[],
                uncertainty=[
                    "No se recuperaron fragmentos de evidencia."
                ],
                citations=[],
                human_review=True,
                abstained=True,
            )

            decision_log.append(
                {
                    "agent": "analyst",
                    "decision": "abstained_no_evidence",
                    "citation_count": 0,
                    "evidence_count": 0,
                }
            )

            return {
                "answer": answer,
                "decision_log": decision_log,
            }

        prompt = self._build_prompt(
            question.question,
            retrieved_chunks,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        raw_response = response.choices[0].message.content

        if not raw_response:
            raise ValueError("Analyst LLM returned an empty response.")

        answer = self._parse_response(
            raw_response,
            question.question_id,
        )

        decision_log.append(
            {
                "agent": "analyst",
                "decision": "generated_structured_answer",
                "citation_count": len(answer.citations),
                "evidence_count": len(retrieved_chunks),
            }
        )

        return {
            "answer": answer,
            "decision_log": decision_log,
        }
