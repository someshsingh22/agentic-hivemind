import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node required for pi extension smoke")
class SubmitTests(unittest.TestCase):
    def test_invalid_dependencies_are_rejected_before_write_and_can_be_corrected(self):
        version = subprocess.check_output(["node", "--version"], text=True)
        if int(version.lstrip("v").split(".")[0]) < 24:
            self.skipTest("Node 24+ required for native TypeScript")
        package = Path(__file__).resolve().parents[1] / "ssh"
        schema = json.loads((package / "schema.json").read_text())
        schema["properties"]["features"]["items"]["properties"]["source_columns"]["items"]["enum"] = ["argument", "title"]
        base = dict(name="base", dtype="binary", source_columns=["argument", "title"],
                    definition="Base measurement.", kind="prompt", spec="Return binary label.",
                    annotation_unit="row", depends_on=[], **{"class": "primary"})
        derived = dict(base, name="derived", kind="code", source_columns=["argument"],
                       depends_on=["base"], **{"class": "secondary"})
        payload = {"features": [base, derived]}
        script = """
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
const { default: extension } = await import(process.env.EXTENSION_URL);
let tool;
extension({ registerTool(value) { tool = value; } });
const payload = JSON.parse(process.env.PAYLOAD);
await assert.rejects(tool.execute('bad', payload), /source_columns must cover/);
assert.equal(existsSync(process.env.SUBMIT_OUT), false);
payload.features[1].source_columns = ['argument', 'title'];
payload.features[1].depends_on = [];
await assert.rejects(tool.execute('bad-class', payload), /secondary features must/);
assert.equal(existsSync(process.env.SUBMIT_OUT), false);
payload.features[1].depends_on = ['base'];
const result = await tool.execute('corrected', payload);
assert.equal(result.terminate, true);
assert.deepEqual(JSON.parse(readFileSync(process.env.SUBMIT_OUT, 'utf8')), payload);
console.log('Submission smoke passed: both reported contract failures rejected before write; corrected submission saved.');
"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            schema_path = root / "schema.json"
            schema_path.write_text(json.dumps(schema))
            result = subprocess.run(["node", "--input-type=module", "-e", script], check=True,
                                    capture_output=True, text=True, env={**os.environ,
                "SUBMIT_SCHEMA": str(schema_path), "SUBMIT_OUT": str(root / "output.json"),
                "SUBMIT_PYTHON": sys.executable, "SUBMIT_FEATURE_VALIDATOR": str(package / "features.py"),
                "EXTENSION_URL": (package / "submit.ts").as_uri(), "PAYLOAD": json.dumps(payload),
            })
            print(result.stdout.strip())
