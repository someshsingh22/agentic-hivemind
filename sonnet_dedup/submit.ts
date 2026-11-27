import { readFileSync, writeFileSync } from "node:fs";

export default function (pi: { registerTool(tool: object): void }) {
	const schema = JSON.parse(readFileSync(process.env.SUBMIT_SCHEMA!, "utf8"));
	const allowed = new Set<string>(schema.properties.groups.items.properties.members.items.enum);
	pi.registerTool({
		name: "submit",
		label: "Submit partition",
		description: "Submit every supplied feature exactly once. If rejected, correct the partition and resubmit. Do not rewrite features.",
		parameters: schema,
		async execute(_id: string, params: { groups: { members: string[]; reason: string }[] }) {
			const seen = new Set<string>();
			if (!Array.isArray(params.groups) || params.groups.length === 0) {
				throw new Error("Return a nonempty groups array. Nothing was saved.");
			}
			for (const group of params.groups) {
				if (!Array.isArray(group.members) || group.members.length === 0 || typeof group.reason !== "string" || !group.reason.trim()) {
					throw new Error("Each group needs nonempty members and a reason. Nothing was saved.");
				}
				const owners = new Set<string>();
				for (const member of group.members) {
					if (!allowed.has(member) || seen.has(member)) {
						throw new Error(`Unknown or repeated feature: ${member}. Nothing was saved.`);
					}
					const owner = member.split(":", 1)[0];
					if (owners.has(owner)) {
						throw new Error("Features from the same hypothesis must stay in separate groups. Nothing was saved.");
					}
					owners.add(owner);
					seen.add(member);
				}
			}
			if (seen.size !== allowed.size) {
				throw new Error("Some supplied features are missing. Include every feature exactly once. Nothing was saved.");
			}
			writeFileSync(process.env.SUBMIT_OUT!, JSON.stringify(params, null, 2));
			return { content: [{ type: "text", text: "Submitted." }], details: {}, terminate: true };
		},
	});
}
