#!/usr/bin/env node
/**
 * Import a thoughtbench leaderboard.json into the landing page's chart data.
 *
 *   node scripts/import-thoughtbench.mjs path/to/leaderboard.json
 *
 * Writes lib/autotree-bench-results.json with rows shaped like EngineRow
 * (lib/autotree-bench-data.ts). Only measured values are carried over;
 * anything absent stays absent - the charts render nulls as gaps, never
 * as zeros. Engine attribution comes from meta.engine_label, which must
 * start with "autotree", "vllm", or "sglang" (case-insensitive).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const src = process.argv[2];
if (!src) {
  console.error("usage: node scripts/import-thoughtbench.mjs <leaderboard.json>");
  process.exit(1);
}

const board = JSON.parse(readFileSync(resolve(src), "utf8"));
if (!Array.isArray(board.entries)) {
  console.error("not a thoughtbench leaderboard: missing entries[]");
  process.exit(1);
}

const engineOf = (label) => {
  const l = String(label).toLowerCase();
  if (l.startsWith("autotree")) return "autotree";
  if (l.startsWith("vllm")) return "vllm";
  if (l.startsWith("sglang")) return "sglang";
  return null;
};

const rows = [];
for (const e of board.entries) {
  const engine = engineOf(e.meta?.engine_label);
  if (!engine) {
    console.warn(`skipping entry with unknown engine_label: ${e.meta?.engine_label}`);
    continue;
  }
  const s = e.summary ?? {};
  const m = e.meta ?? {};
  const nTasks = s.n_tasks ?? 0;
  const seeds = Array.isArray(m.seeds) ? m.seeds.length : 0;
  rows.push({
    id: `tb-${engine}-${m.arm}-${(m.model || "model").split("/").pop()}-${e.source || rows.length}`
      .toLowerCase()
      .replace(/[^a-z0-9-]+/g, "-"),
    engine,
    label: `${m.model} ${m.arm}`,
    status: "measured",
    accuracy: typeof s.accuracy === "number" ? s.accuracy * 100 : undefined,
    ciLow: typeof s.ci_low === "number" ? s.ci_low * 100 : undefined,
    ciHigh: typeof s.ci_high === "number" ? s.ci_high * 100 : undefined,
    meanTokens: s.mean_tokens ?? undefined,
    tokensPerCorrect: s.tokens_per_correct ?? undefined,
    regime: `thoughtbench | ${m.model} | arm=${m.arm} | n=${nTasks} tasks x ${seeds} seed(s)${m.git_sha ? ` | ${String(m.git_sha).slice(0, 9)}` : ""}`,
  });
}

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, "..", "lib", "autotree-bench-results.json");
writeFileSync(out, JSON.stringify({ generated_at: board.generated_at ?? null, rows }, null, 2) + "\n");
console.log(`wrote ${rows.length} measured rows -> lib/autotree-bench-results.json`);
