// M5 确认卡片的状态机：纯函数，不碰 DOM，方便 node --test 单测。
// 卡片状态只来自后端（GET /approvals、approval_request / approval_update 事件、决定接口的返回），
// 前端从不自己「猜」下一个状态：两个标签页、重连重放时，以库里的为准。

export const VIEW = {
  pending:         { label: "待确认", buttons: [["approve", "批准"], ["reject", "拒绝"]] },
  approved:        { label: "已批准，等待执行", buttons: [] },
  executing:       { label: "执行中…", buttons: [] },
  succeeded:       { label: "已完成", buttons: [] },
  failed:          { label: "执行失败", buttons: [] },
  rejected:        { label: "已拒绝", buttons: [] },
  superseded:      { label: "已作废（你发了新消息）", buttons: [] },
  denied:          { label: "规则禁止", buttons: [] },
  // 结果未知：不提供「重试」，只让人核实外部状态后告诉模型
  // O10：支持按幂等键查询的工具（如 transfer）多一个「让系统查」：查不出结论时接口回 409，卡片保持原样
  outcome_unknown: { label: "结果未知：请先核实是否已生效", buttons: [["done", "已生效"], ["not_done", "未生效"]] },
  deferred:        { label: "排在待确认的操作后面，批准后按顺序执行", buttons: [] },
};

const RESOLVED = { done: "结果未知 · 你确认已生效", not_done: "结果未知 · 你确认未生效" };
const RECONCILED = { done: "结果未知 · 系统按调用编号查到：已生效", not_done: "结果未知 · 系统按调用编号查到：未生效" };

// 一条审批记录 → {label, buttons}。核实过的结果未知不再给按钮
export function view(a) {
  if (a.status === "outcome_unknown" && a.resolution)
    return { label: (a.resolved_by === "reconcile" ? RECONCILED : RESOLVED)[a.resolution], buttons: [] };
  // O10 复核⑤：只有支持按键查询的工具（列表接口给 reconcilable）才多一个「让系统查」
  if (a.status === "outcome_unknown" && a.reconcilable)
    return { ...VIEW.outcome_unknown, buttons: [["reconcile", "让系统按调用编号查"], ...VIEW.outcome_unknown.buttons] };
  return VIEW[a.status] ?? { label: a.status, buttons: [] };
}

// 决定接口的返回 → 新的 {status, resolution}。200 用返回值；409 用 detail 里的真实状态；其他错误保持原样
export function afterDecision(a, httpStatus, body) {
  if (httpStatus === 200) return { ...a, status: body.status, resolution: body.resolution, resolved_by: body.resolved_by ?? a.resolved_by };
  if (httpStatus === 409 && body?.detail?.status)  // 复核⑥：结论是谁给的也要跟着真实状态走
    return { ...a, status: body.detail.status, resolution: body.detail.resolution, resolved_by: body.detail.resolved_by ?? a.resolved_by };
  return a;
}

// 事件 → 更新审批表（Map: tool_use_id -> 记录）。返回新 Map，不改入参
export function applyApprovalEvent(map, ev) {
  const m = new Map(map);
  if (ev.type === "approval_request") {
    for (const a of ev.approvals) m.set(a.tool_use_id, { ...a, decision: "ask", resolution: null });
  } else if (ev.type === "approval_update" && m.has(ev.tool_use_id)) {
    m.set(ev.tool_use_id, { ...m.get(ev.tool_use_id), status: ev.status });
  }
  return m;
}
