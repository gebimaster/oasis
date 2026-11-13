// M7 子任务胶囊的状态：纯函数，不碰 DOM，方便 node --test 单测。
// 两个来源：父 run 流里的 subagent_update（实时），和 GET /subagents（刷新后以库为准）。key = tool_use_id#index

export const LABEL = { queued: "排队中", running: "运行中", completed: "完成", failed: "失败",
                       timeout: "超时", cancelled: "已取消", interrupted: "已中断" };
export const key = (toolUseId, index) => `${toolUseId}#${index}`;

export function applySubEvent(map, ev) {
  const next = new Map(map);
  const k = key(ev.tool_use_id, ev.index);
  const old = map.get(k) || {};
  // 终态事件不带 child_run_id，沿用 running 时记下的那个
  next.set(k, { ...old, title: ev.title, status: ev.status, runId: ev.child_run_id || old.runId });
  return next;
}

// 库里的 run 状态 → 胶囊状态：超时和中断在库里是 failed，靠 stop_reason 区分
export function statusOfRun(r) {
  if (r.status === "failed" && r.stop_reason === "timeout") return "timeout";
  if (r.status === "failed" && r.stop_reason === "interrupted") return "interrupted";
  return r.status;
}

export function fromRows(rows) {
  return new Map(rows.map((r) => [key(r.tool_use_id, r.index),
                                  { title: r.title, status: statusOfRun(r), runId: r.run_id }]));
}

// 某次 spawn_subagents 的胶囊，按 index 排好
export const capsulesOf = (map, toolUseId) =>
  [...map.entries()].filter(([k]) => k.startsWith(toolUseId + "#"))
    .sort(([a], [b]) => Number(a.split("#")[1]) - Number(b.split("#")[1])).map(([, v]) => v);
