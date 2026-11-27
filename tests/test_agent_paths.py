import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ssh.decompose import decompose_all


class AgentPathTests(unittest.TestCase):
    def test_relative_decomposition_paths_work_from_pi_session_directory(self):
        with tempfile.TemporaryDirectory(prefix="ssh path regression ") as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            executable = binary / "pi"
            result = {"features": [{"name": "uncertainty", "dtype": "binary",
                "source_columns": ["argument"], "definition": "Explicit uncertainty",
                "kind": "prompt", "spec": "Return 1 if uncertain, 0 otherwise, null if missing"}],
                "h0": "No association", "h1": "Positive association", "rhs": "uncertainty",
                "term": "uncertainty", "sign": "+", "note": ""}
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\nfrom pathlib import Path\n"
                "schema_path = Path(os.environ['SUBMIT_SCHEMA'])\n"
                "out = Path(os.environ['SUBMIT_OUT'])\n"
                "assert schema_path.is_absolute() and out.is_absolute()\n"
                "schema = json.loads(schema_path.read_text())\n"
                "assert schema['properties']['features']['items']['properties']['source_columns']['items']['enum'] == ['argument']\n"
                "assert Path(sys.argv[sys.argv.index('-e') + 1]).is_file()\n"
                "assert Path(os.environ['SUBMIT_FEATURE_VALIDATOR']).is_file()\n"
                "assert Path.cwd() == Path(os.environ['PI_CODING_AGENT_DIR'])\n"
                "assert (Path.cwd() / 'models.json').is_file()\n"
                f"out.write_text(json.dumps({result!r}))\n"
            )
            executable.chmod(0o755)
            source = root / "data.csv"
            source.write_text("argument,delta,post_id\nhello,0,a\n")
            out = Path(os.path.relpath(root / "runs/cmv", Path.cwd()))
            cfg = {"model": "unused", "max_tokens": 100, "compat": {},
                   "base_url": "https://unused.invalid", "api": "anthropic-messages",
                   "api_key_env": "UNUSED_KEY", "provider": "unused", "workers": 1,
                   "thinking": "off", "tools": "submit"}
            env = {"csv": str(source), "target": "delta", "group": "post_id"}
            with patch.dict(os.environ, {"PATH": str(binary) + os.pathsep + os.environ["PATH"]}):
                actual = decompose_all({0: "Uncertainty predicts delta"}, env, cfg, out)
            self.assertEqual(actual[0]["features"], result["features"])
            self.assertEqual(json.loads((out / "decomposed/0.json").read_text()), result)
