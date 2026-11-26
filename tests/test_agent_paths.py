import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ssh.equivalence import EquivalenceReviewer


class AgentPathTests(unittest.TestCase):
    def test_relative_review_paths_work_from_pi_session_directory(self):
        with tempfile.TemporaryDirectory(prefix="ssh path regression ") as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            executable = binary / "pi"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "schema = json.loads(Path(os.environ['SUBMIT_SCHEMA']).read_text())\n"
                "assert schema['properties']['comparisons']['minItems'] == 1\n"
                "extension = Path(sys.argv[sys.argv.index('-e') + 1])\n"
                "assert extension.is_file()\n"
                "assert Path.cwd() == Path(os.environ['PI_CODING_AGENT_DIR'])\n"
                "Path(os.environ['SUBMIT_OUT']).write_text(json.dumps({'comparisons': ["
                "{'left': 0, 'right': 1, 'verdict': 'distinct', 'reason': 'Any differs from all'}]}))\n"
            )
            executable.chmod(0o755)
            relative_out = Path(os.path.relpath(root / "calibration/cmv", Path.cwd()))
            cfg = {"agent": {
                "model": "unused", "max_tokens": 100, "compat": {}, "base_url": "https://unused.invalid",
                "api": "anthropic-messages", "api_key_env": "UNUSED_KEY", "provider": "unused",
                "workers": 1, "thinking": "off", "tools": "submit",
            }}
            features = [{"hypothesis_id": i, "feature": {
                "name": name, "dtype": "binary", "source_columns": ["OC_comments"],
                "definition": name, "kind": "prompt", "spec": name,
            }} for i, name in enumerate(["any_comment", "all_comments"])]
            with patch.dict(os.environ, {"PATH": str(binary) + os.pathsep + os.environ["PATH"]}):
                result = EquivalenceReviewer(cfg, relative_out)(features)
            self.assertEqual(result, [{"left": 0, "right": 1, "verdict": "distinct", "reason": "Any differs from all"}])
            saved = [p for p in (root / "calibration/cmv/dedup_reviews").glob("*.json")
                     if not p.name.endswith((".inputs.json", ".schema.json"))]
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text()), {"comparisons": result})


if __name__ == "__main__":
    unittest.main()
