"""Validate scalar row measurements without executing their specifications."""

import json
import re
import sys
from pathlib import Path


_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_FIELDS = {"name", "dtype", "source_columns", "definition", "kind", "spec"}


def validate_features(features, columns=None):
    """Return features unchanged, or raise ValueError for an invalid contract.

    ``columns``, when supplied, is the allowed original predictor-column collection
    (already excluding outcomes and identifiers); source metadata never contains
    generated feature names.
    """
    if not isinstance(features, list):
        raise ValueError("Features must be a list")
    allowed = None if columns is None else set(columns)
    names = set()
    for feature in features:
        if not isinstance(feature, dict) or set(feature) != _FIELDS:
            raise ValueError("Each feature needs exactly name, dtype, source_columns, definition, kind and spec")
        name = feature["name"]
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("Feature names must be snake_case identifiers")
        if name in names:
            raise ValueError(f"Duplicate feature name: {name}")
        for field, values in (
            ("dtype", ("binary", "numeric")),
            ("kind", ("code", "prompt")),
        ):
            if feature[field] not in values:
                raise ValueError(f"{name}: invalid {field}")
        for field in ("definition", "spec"):
            if not isinstance(feature[field], str) or not feature[field].strip():
                raise ValueError(f"{name}: {field} must be a nonempty string")
        sources = feature["source_columns"]
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(source, str) or not source.strip() or source != source.strip() for source in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError(f"{name}: source_columns must contain unique, nonempty original column names")
        if set(sources) & {"post_id", "comment_id"}:
            raise ValueError(f"{name}: identifier columns are not measurement sources")
        if allowed is not None and not set(sources) <= allowed:
            raise ValueError(f"{name}: source_columns must be available original predictor columns")
        names.add(name)

    return features


if __name__ == "__main__":
    schema = json.loads(Path(sys.argv[1]).read_text())
    columns = schema["properties"]["features"]["items"]["properties"]["source_columns"]["items"].get("enum")
    try:
        validate_features(json.load(sys.stdin)["features"], columns)
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
