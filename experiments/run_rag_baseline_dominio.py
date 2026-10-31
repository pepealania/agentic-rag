import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_baseline_dominio import RAGBaseline, load_domain_config


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}:{line_number}"
                )

            records.append(record)

    return records


def main() -> None:
    config = load_domain_config()
    corpus_dir = ROOT / config["paths"]["raw_data"]
    questions_path = ROOT / config["evaluation"]["questions_path"]
    output_dir = ROOT / config["paths"]["outputs"]

    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Domain corpus directory does not exist: {corpus_dir}"
        )

    if not questions_path.is_file():
        raise FileNotFoundError(
            f"Domain questions file does not exist: {questions_path}"
        )

    questions = load_jsonl(questions_path)
    if not questions:
        raise ValueError(f"No evaluation questions found in {questions_path}")

    rag = RAGBaseline(config)

    documents = rag.load_documents(corpus_dir)
    print(f"Documents: {len(documents)}")

    chunks = rag.build_chunks()
    print(f"Chunks: {len(chunks)}")

    rag.build_index()
    print("Index ready.")

    results = []
    for question in questions:
        result = rag.run(
            question["question_id"],
            question["question"],
        )
        result["reference_answer"] = question.get("reference_answer", "")
        result["expected_evidence_ids"] = question.get(
            "expected_evidence_ids", []
        )
        result["scenario"] = question.get("scenario", "")
        results.append(result)

        print(
            question["question_id"],
            "|",
            result["json_valid"],
            "|",
            result["citation_validity"],
            "|",
            result["latency_seconds"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)

    csv_path = output_dir / "results.csv"
    fieldnames = list(dict.fromkeys(
        key for result in results for key in result
    ))
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in result.items()
                }
            )

    print(f"Results JSON: {json_path}")
    print(f"Results CSV: {csv_path}")


if __name__ == "__main__":
    main()