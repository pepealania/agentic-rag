import json
import os
import pandas as pd

from src.config import load_config
from src.rag_baseline import RAGBaseline


CONFIG_PATH = "configs/default.yaml"

config = load_config(CONFIG_PATH)

rag = RAGBaseline(config)

rag.load_documents(
    config["paths"]["raw_data"]
)

print(
    "Documents:",
    len(rag.documents)
)

rag.build_chunks()

print(
    "Chunks:",
    len(rag.chunks)
)

rag.build_index()

print("Index ready.")

questions_path = config[
    "evaluation"
]["questions_path"]

questions = []

with open(
    questions_path,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():
            questions.append(
                json.loads(line)
            )

results = []

for question in questions:

    result = rag.run(
        question["question_id"],
        question["question"]
    )

    result["reference_answer"] = (
        question.get(
            "reference_answer",
            ""
        )
    )

    results.append(result)

    print(
        question["question_id"],
        "|",
        result["json_valid"],
        "|",
        result["citation_validity"],
        "|",
        result["latency_seconds"]
    )

os.makedirs(
    "outputs/rag_baseline",
    exist_ok=True
)

df = pd.DataFrame(results)

df.to_json(
    "outputs/rag_baseline/results.json",
    orient="records",
    force_ascii=False,
    indent=2
)

df.to_csv(
    "outputs/rag_baseline/results.csv",
    index=False
)

print("\nFinished.")
print(df)
