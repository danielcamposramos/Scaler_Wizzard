"""Runtime JSON schema validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Optional dependency
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore


DEFAULT_SCHEMA = Path(__file__).resolve().parents[2] / "specs" / "cockpit_schema.json"


@dataclass
class ValidationResult:
    ok: bool
    errors: Optional[list[str]] = None


class SchemaValidator:
    def __init__(self, schema_path: Path = DEFAULT_SCHEMA):
        self.schema_path = schema_path
        self._schema_cache: Optional[Dict[str, Any]] = None

    def load_schema(self) -> Dict[str, Any]:
        if self._schema_cache is not None:
            return self._schema_cache
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {self.schema_path}")
        with self.schema_path.open("r", encoding="utf-8") as handle:
            self._schema_cache = json.load(handle)
        return self._schema_cache

    def validate(self, payload: Dict[str, Any]) -> ValidationResult:
        if jsonschema is None:
            # Soft validation when dependency missing
            return ValidationResult(ok=True, errors=["jsonschema library not installed; validation skipped"])
        schema = self.load_schema()
        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
        if not errors:
            return ValidationResult(ok=True)
        messages = [f"{'.'.join(map(str, err.path))}: {err.message}" for err in errors]
        return ValidationResult(ok=False, errors=messages)


__all__ = ["SchemaValidator", "ValidationResult"]
