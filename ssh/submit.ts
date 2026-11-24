// pi extension: a terminating `submit` tool whose arguments must match the JSON schema at $SUBMIT_SCHEMA.
// pi validates the call (errors go back to the model) and the result is written to $SUBMIT_OUT.
import { readFileSync, writeFileSync } from "node:fs";

export default function (pi: any) {
	pi.registerTool({
		name: "submit",
		label: "Submit",
		description: "Submit the final answer. Call exactly once, as your last action.",
		parameters: JSON.parse(readFileSync(process.env.SUBMIT_SCHEMA!, "utf8")),
		async execute(_id: string, params: unknown) {
			writeFileSync(process.env.SUBMIT_OUT!, JSON.stringify(params, null, 2));
			return { content: [{ type: "text", text: "Submitted." }], details: {}, terminate: true };
		},
	});
}
