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
    """Represents the result of a schema validation.

    Attributes:
        ok (bool): True if the validation was successful, False otherwise.
        errors (Optional[list[str]]): A list of validation error messages,
                                      if any.
    """
    ok: bool
    errors: Optional[list[str]] = None


class SchemaValidator:
    """A validator for checking JSON payloads against a JSON schema.

    This class provides methods to load a schema and validate dictionaries
    against it. It gracefully handles the absence of the `jsonschema` library
    by skipping validation.
    """
    def __init__(self, schema_path: Path = DEFAULT_SCHEMA):
        """Initializes the SchemaValidator.

        Args:
            schema_path (Path): The path to the JSON schema file. Defaults to
                                the main cockpit schema.
        """
        self.schema_path = schema_path
        self._schema_cache: Optional[Dict[str, Any]] = None

    def load_schema(self) -> Dict[str, Any]:
        """Loads the JSON schema from the specified path.

        The loaded schema is cached in memory to avoid repeated file reads.

        Returns:
            Dict[str, Any]: The loaded JSON schema as a dictionary.

        Raises:
            FileNotFoundError: If the schema file cannot be found.
        """
        if self._schema_cache is not None:
            return self._schema_cache
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {self.schema_path}")
        with self.schema_path.open("r", encoding="utf-8") as handle:
            self._schema_cache = json.load(handle)
        return self._schema_cache

    def validate(self, payload: Dict[str, Any]) -> ValidationResult:
        """Validates a payload against the loaded JSON schema.

        If the `jsonschema` library is not installed, validation is skipped,
        and a successful result is returned with a warning.

        Args:
            payload (Dict[str, Any]): The JSON payload to validate.

        Returns:
            ValidationResult: An object containing the validation status and
                              a list of errors, if any.
        """
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
