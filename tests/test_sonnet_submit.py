import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node required for the Sonnet submit extension")
class SonnetSubmitTests(unittest.TestCase):
    def test_incomplete_or_inconsistent_partition_is_not_saved_and_can_be_corrected(self):
        version = subprocess.check_output(["node", "--version"], text=True)
        if int(version.lstrip("v").split(".")[0]) < 24:
            self.skipTest("Node 24+ required for native TypeScript")
        package = Path(__file__).resolve().parents[1] / "sonnet_dedup"
        schema = json.loads((package / "schema.json").read_text())
        keys = ["h00001:a", "h00001:b", "h00002:c"]
        schema["properties"]["groups"]["items"]["properties"]["members"]["items"]["enum"] = keys
        script = """
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
const { default: extension } = await import(process.env.EXTENSION_URL);
let tool;
extension({ registerTool(value) { tool = value; } });
const group = members => ({ members, reason: 'Same measurement or retained separately.' });
for (const groups of [
    [group(['h00001:a', 'h00002:c'])],
    [group(['h00001:a']), group(['h00001:a', 'h00002:c']), group(['h00001:b'])],
    [group(['h00001:a', 'h00001:b']), group(['h00002:c'])],
    [group(['h00001:a']), group(['h00001:b']), group(['h99999:unknown'])],
]) {
    await assert.rejects(tool.execute('invalid', { groups }));
    assert.equal(existsSync(process.env.SUBMIT_OUT), false);
}
const valid = { groups: [group(['h00001:a', 'h00002:c']), group(['h00001:b'])] };
const result = await tool.execute('corrected', valid);
assert.equal(result.terminate, true);
assert.deepEqual(JSON.parse(readFileSync(process.env.SUBMIT_OUT, 'utf8')), valid);
console.log('Sonnet partition submission: omissions, repetition, unknown keys and same-hypothesis merges rejected before saving; corrected partition accepted.');
"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            schema_path = root / "schema.json"
            schema_path.write_text(json.dumps(schema))
            result = subprocess.run(["node", "--input-type=module", "-e", script], check=True,
                                    capture_output=True, text=True, env={**os.environ,
                "SUBMIT_SCHEMA": str(schema_path), "SUBMIT_OUT": str(root / "response.json"),
                "EXTENSION_URL": (package / "submit.ts").as_uri(),
            })
            print(result.stdout.strip())
