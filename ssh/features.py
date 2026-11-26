"""Validate reusable measurement contracts without executing their specifications."""

import json
import re
import sys
from collections import deque
from pathlib import Path


_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_FIELDS = {
    "name", "dtype", "source_columns", "definition", "kind", "spec",
    "class", "annotation_unit", "depends_on",
}


def validate_features(features, columns=None):
    """Return features unchanged, or raise ValueError for an invalid contract.

    Dependencies name features in this same list. ``columns``, when supplied, is
    the allowed original predictor-column collection (already excluding outcomes
    and identifiers); source metadata never contains generated feature names.
    Comment outputs align to OC_comments, optionally reading title/argument as
    context, rather than supporting arbitrary list-valued predictor columns.
    """
    if not isinstance(features, list):
        raise ValueError("Features must be a list")
    allowed = None if columns is None else set(columns)
    by_name = {}
    for feature in features:
        if not isinstance(feature, dict) or set(feature) != _FIELDS:
            raise ValueError("Each feature needs exactly name, dtype, source_columns, definition, kind, spec, class, annotation_unit and depends_on")
        name = feature["name"]
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("Feature names must be snake_case identifiers")
        if name in by_name:
            raise ValueError(f"Duplicate feature name: {name}")
        for field, values in (
            ("dtype", ("binary", "numeric")),
            ("kind", ("code", "prompt")),
            ("class", ("primary", "secondary")),
            ("annotation_unit", ("row", "comment")),
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
        deps = feature["depends_on"]
        if (not isinstance(deps, list)
                or any(not isinstance(dep, str) or not _NAME.fullmatch(dep) for dep in deps)
                or len(set(deps)) != len(deps)):
            raise ValueError(f"{name}: depends_on must contain unique feature names")
        if feature["class"] == "primary" and deps:
            raise ValueError(f"{name}: primary features must have empty depends_on")
        if feature["class"] == "secondary" and (
            feature["kind"] != "code" or feature["annotation_unit"] != "row" or not deps
        ):
            raise ValueError(f"{name}: secondary features must be code, row-level and depend on base features")
        if feature["annotation_unit"] == "comment" and (
            "OC_comments" not in sources or not set(sources) <= {"OC_comments", "title", "argument"}
        ):
            raise ValueError(f"{name}: comment annotation requires OC_comments with only optional title/argument context")
        by_name[name] = feature

    dependents = {name: [] for name in by_name}
    pending = {}
    for name, feature in by_name.items():
        deps = feature["depends_on"]
        if any(dep not in by_name for dep in deps):
            raise ValueError(f"{name}: depends_on references an unknown feature")
        required_sources = {source for dep in deps for source in by_name[dep]["source_columns"]}
        if not required_sources <= set(feature["source_columns"]):
            raise ValueError(f"{name}: source_columns must cover all original sources of dependencies")
        pending[name] = len(deps)
        for dep in deps:
            dependents[dep].append(name)
    ready = deque(name for name, count in pending.items() if count == 0)
    visited = 0
    while ready:
        visited += 1
        for name in dependents[ready.popleft()]:
            pending[name] -= 1
            if pending[name] == 0:
                ready.append(name)
    if visited != len(features):
        raise ValueError("Feature dependencies must be acyclic")
    return features


if __name__ == "__main__":
    schema = json.loads(Path(sys.argv[1]).read_text())
    columns = schema["properties"]["features"]["items"]["properties"]["source_columns"]["items"].get("enum")
    try:
        validate_features(json.load(sys.stdin)["features"], columns)
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
