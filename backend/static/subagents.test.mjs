import { test } from "node:test";
import assert from "node:assert/strict";
import { applySubEvent, capsulesOf, fromRows } from "./subagents.js";

test("live events: queued → running keeps run id through the final status", () => {
  let m = new Map();
  m = applySubEvent(m, { tool_use_id: "sp1", index: 1, title: "乙", status: "queued" });
  m = applySubEvent(m, { tool_use_id: "sp1", index: 0, title: "甲", status: "running", child_run_id: "r0" });
  m = applySubEvent(m, { tool_use_id: "sp1", index: 0, title: "甲", status: "completed" });
  assert.deepEqual(capsulesOf(m, "sp1"), [{ title: "甲", status: "completed", runId: "r0" },
                                          { title: "乙", status: "queued", runId: undefined }]);
});

test("rows from db: timeout and interrupted are told apart by stop_reason", () => {
  const m = fromRows([
    { tool_use_id: "sp1", index: 0, title: "甲", status: "failed", stop_reason: "timeout", run_id: "r0" },
    { tool_use_id: "sp1", index: 1, title: "乙", status: "failed", stop_reason: "interrupted", run_id: "r1" },
    { tool_use_id: "sp2", index: 0, title: "丙", status: "cancelled", stop_reason: "parent_cancel", run_id: "r2" },
  ]);
  assert.deepEqual(capsulesOf(m, "sp1").map((c) => c.status), ["timeout", "interrupted"]);
  assert.deepEqual(capsulesOf(m, "sp2").map((c) => c.status), ["cancelled"]);
});
