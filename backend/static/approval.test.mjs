// node --test static/approval.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { afterDecision, applyApprovalEvent, view } from "./approval.js";

const pending = { tool_use_id: "p1", status: "pending", resolution: null };

test("待确认有批准/拒绝，结果未知只有核实按钮", () => {
  assert.deepEqual(view(pending).buttons.map((b) => b[0]), ["approve", "reject"]);
  assert.deepEqual(view({ status: "outcome_unknown", resolution: null }).buttons.map((b) => b[0]), ["done", "not_done"]);
  assert.deepEqual(view({ status: "outcome_unknown", resolution: null, reconcilable: true }).buttons.map((b) => b[0]),
                   ["reconcile", "done", "not_done"]);
  const after409 = afterDecision({ status: "outcome_unknown" }, 409,
                                 { detail: { status: "outcome_unknown", resolution: "done", resolved_by: "reconcile" } });
  assert.match(view(after409).label, /系统按调用编号查到/);
  assert.match(view({ status: "outcome_unknown", resolution: "done", resolved_by: "reconcile" }).label, /系统按调用编号查到/);
  assert.equal(view({ status: "outcome_unknown", resolution: "done" }).buttons.length, 0);
});

test("409 时卡片切到后端给的真实状态，而不是停在待确认", () => {
  const a = afterDecision(pending, 409, { detail: { error: "状态已经变了", status: "superseded", resolution: null } });
  assert.equal(a.status, "superseded");
  assert.equal(afterDecision(pending, 500, {}), pending);         // 网络/服务错误：不乱改，允许再点
});

test("事件驱动：approval_request 建卡，approval_update 推进，未知卡片的更新被忽略", () => {
  let m = applyApprovalEvent(new Map(), { type: "approval_request", approvals: [{ tool_use_id: "p1", status: "pending" }] });
  m = applyApprovalEvent(m, { type: "approval_update", tool_use_id: "p1", status: "executing" });
  m = applyApprovalEvent(m, { type: "approval_update", tool_use_id: "zz", status: "succeeded" });
  assert.equal(m.get("p1").status, "executing");
  assert.equal(m.has("zz"), false);
});
