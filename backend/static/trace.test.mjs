// M8：node --test static/trace.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { bar, bounds, flatten, summary } from "./trace.js";

const T = (s) => `2026-10-30T10:00:${String(s).padStart(2, "0")}.000000+00:00`;  // 后端给的是微秒精度

test("flatten 深度优先、带层级", () => {
  const roots = [{ id: "r", children: [{ id: "a", children: [{ id: "a1", children: [] }] }, { id: "b", children: [] }] }];
  assert.deepEqual(flatten(roots).map((x) => [x.id, x.depth]), [["r", 0], ["a", 1], ["a1", 2], ["b", 1]]);
});

test("瀑布条按整棵树的时间范围换算，没结束的算到 now", () => {
  const rows = [{ started_at: T(0), ended_at: T(10) }, { started_at: T(5), ended_at: null }];
  const now = Date.parse(T(20));
  const b = bounds(rows, now);
  assert.equal(b.end - b.start, 20000);
  assert.deepEqual(bar(rows[1], b, now), { left: 25, width: 75 });
  assert.equal(bar({ started_at: T(1), ended_at: T(1) }, b, now).width, 0.5);  // 0 毫秒也看得见
});

test("summary 按 kind 给摘要", () => {
  const llm = { kind: "llm", attrs: { stop_reason: "tool_use", usage: { input_tokens: 9, output_tokens: 3 } } };
  assert.equal(summary(llm), "入 9 / 缓存 0 / 出 3 · tool_use");
  assert.equal(summary({ kind: "tool", attrs: {}, error: "[TOOL_ERROR] unknown tool 'get_stock'" }), "[TOOL_ERROR] unknown tool 'get_stock'");
});
