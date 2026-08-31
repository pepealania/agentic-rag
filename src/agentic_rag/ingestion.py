from __future__ import annotations

from pathlib import Path

from .schemas import Document


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(data_dir: Path) -> list[Document]:
    """Load supported documents from a directory."""

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    documents: list[Document] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        document_id = path.relative_to(data_dir).with_suffix("").as_posix()

        documents.append(
            Document(
                document_id=document_id,
                source=path.relative_to(data_dir).as_posix(),
                text=text,
            )
        )

    return documents
