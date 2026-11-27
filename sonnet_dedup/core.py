"""Validate prompt partitions and assemble a frozen Sonnet-only feature plan."""

from ssh.dedup import FIELDS, rename
from ssh.features import validate_features


def prepare_groups(decomposed, env):
    """Validate the full feature bank, then batch prompts by source set and dtype."""
    forbidden = {env["target"], env.get("group"), "post_id", "comment_id"} - {None}
    owners = set()
    for decomposition in decomposed:
        owner = decomposition["id"]
        if type(owner) is not int or owner in owners:
            raise ValueError("Hypothesis IDs must be unique integers")
        owners.add(owner)
        validate_features(decomposition["features"])
        if any(forbidden.intersection(feature["source_columns"])
               for feature in decomposition["features"]):
            raise ValueError("Outcome and identifier columns cannot be feature sources")
    raw = [(d["id"], feature) for d in decomposed for feature in d["features"]]
    original = forbidden | {source for _, feature in raw for source in feature["source_columns"]}
    if any(feature["name"] in original for _, feature in raw):
        raise ValueError("Feature names cannot collide with original input columns")

    groups = {}
    for owner, feature in raw:
        if feature["kind"] == "prompt":
            batch = (tuple(sorted(feature["source_columns"])), feature["dtype"])
            groups.setdefault(batch, []).append({
                "key": f"h{owner:05d}:{feature['name']}",
                "hypothesis_id": owner,
                "feature": feature,
            })
    return [groups[batch] for batch in sorted(groups)]


def validate_partition(entries, response):
    """Require an exhaustive partition with compatible, distinct-owner members."""
    if (not isinstance(response, dict) or set(response) != {"groups"}
            or not isinstance(response["groups"], list)):
        raise ValueError("Partition response must contain a groups list")
    by_key = {entry["key"]: entry for entry in entries}
    seen = set()
    for group in response["groups"]:
        if not isinstance(group, dict) or set(group) != {"members", "reason"}:
            raise ValueError("Each group must contain members and reason")
        if not isinstance(group["reason"], str) or not group["reason"].strip():
            raise ValueError("Group reason must be a nonempty string")
        members = group["members"]
        if not isinstance(members, list) or not members:
            raise ValueError("Groups must have nonempty member lists")
        owners = set()
        measurement = None
        for key in members:
            if not isinstance(key, str) or key not in by_key:
                raise ValueError(f"Unknown feature key: {key!r}")
            if key in seen:
                raise ValueError(f"Repeated feature key: {key}")
            seen.add(key)
            entry = by_key[key]
            if entry["hypothesis_id"] in owners:
                raise ValueError("A merged group cannot repeat a hypothesis")
            owners.add(entry["hypothesis_id"])
            feature = entry["feature"]
            signature = (tuple(sorted(feature["source_columns"])), feature["dtype"])
            if measurement is not None and signature != measurement:
                raise ValueError("Merged features must have identical source sets and dtype")
            measurement = signature
    if seen != set(by_key):
        raise ValueError(f"Partition omits feature keys: {sorted(set(by_key) - seen)}")


def build_plan(decomposed, env, partitions):
    """Apply a complete prompt partition without rewriting frozen measurements."""
    entries = [entry for batch in prepare_groups(decomposed, env) for entry in batch]
    validate_partition(entries, {"groups": partitions})
    labels = {key: index for index, group in enumerate(partitions) for key in group["members"]}
    raw = [(d["id"], feature) for d in decomposed for feature in d["features"]]
    original = {env["target"], env.get("group"), "post_id", "comment_id"} - {None}
    original.update(source for _, feature in raw for source in feature["source_columns"])
    canon, taken, mappings = {}, set(original), {d["id"]: {} for d in decomposed}
    for index, (owner, feature) in enumerate(raw):
        label = labels[f"h{owner:05d}:{feature['name']}"] if feature["kind"] == "prompt" else -index - 1
        if label not in canon:
            name = feature["name"]
            suffix = 2
            while name in taken:
                name = f"{feature['name']}_{suffix}"
                suffix += 1
            taken.add(name)
            canon[label] = {**{key: feature[key] for key in FIELDS}, "name": name, "members": []}
        canon[label]["members"].append({"hypothesis": owner, **{key: feature[key] for key in FIELDS}})
        mappings[owner][feature["name"]] = canon[label]["name"]
    for feature in canon.values():
        if feature["kind"] == "code":
            feature["spec"] = rename(feature["spec"], mappings[feature["members"][0]["hypothesis"]])

    hypotheses = [
        {
            "id": d["id"],
            "hypothesis": d["hypothesis"],
            "h0": rename(d["h0"], mappings[d["id"]]),
            "h1": rename(d["h1"], mappings[d["id"]]),
            "formula": f"{env['target']} ~ {rename(d['rhs'], mappings[d['id']])}",
            "term": d["term"] and rename(d["term"], mappings[d["id"]]),
            "sign": d["sign"],
            "note": d["note"],
        }
        for d in decomposed
    ]
    return {
        "env": {key: env[key] for key in ("name", "target", "group")},
        "features": list(canon.values()),
        "hypotheses": hypotheses,
        "partitions": partitions,
    }
