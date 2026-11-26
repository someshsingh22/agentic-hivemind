// pi extension: a terminating `submit` tool whose arguments must match the JSON schema at $SUBMIT_SCHEMA.
// pi validates the call (errors go back to the model) and the result is written to $SUBMIT_OUT.
import { readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

export default function (pi: any) {
	const schema = JSON.parse(readFileSync(process.env.SUBMIT_SCHEMA!, "utf8"));
	pi.registerTool({
		name: "submit",
		label: "Submit",
		description: "Submit the final answer as your last action. If rejected, correct the reported contract errors and resubmit.",
		parameters: schema,
		async execute(_id: string, params: unknown) {
			if (schema.properties?.features) {
				const validation = spawnSync(process.env.SUBMIT_PYTHON!, [
					process.env.SUBMIT_FEATURE_VALIDATOR!,
					process.env.SUBMIT_SCHEMA!,
				], { input: JSON.stringify(params), encoding: "utf8" });
				if (validation.error || validation.status !== 0) {
					throw new Error("Feature contract rejected; correct the feature metadata/specification and resubmit. Nothing was saved.\n"
						+ (validation.error?.message ?? validation.stderr));
				}
			}
			writeFileSync(process.env.SUBMIT_OUT!, JSON.stringify(params, null, 2));
			return { content: [{ type: "text", text: "Submitted." }], details: {}, terminate: true };
		},
	});
}
