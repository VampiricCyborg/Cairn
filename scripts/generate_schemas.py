"""Regenerate cairn/schema/*.json from the Pydantic models in cairn.core.models.

Run this after changing `Entry`, `SessionTrace`, or any model they reference:

    uv run python scripts/generate_schemas.py

`tests/test_models.py` fails in CI if the checked-in schema files fall out of
sync with the models, so this script is how you fix that failure.
"""

import json
from pathlib import Path

from cairn.core.models import entry_json_schema, trace_json_schema

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "cairn" / "schema"


def _write(name: str, schema: dict[str, object]) -> None:
    path = SCHEMA_DIR / name
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    _write("entry.schema.json", entry_json_schema())
    _write("trace.schema.json", trace_json_schema())


if __name__ == "__main__":
    main()
