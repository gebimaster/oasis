// M8：trace 查看页的纯函数（node --test 覆盖）。缩进树 + 瀑布条画在同一行：左边看父子，右边看时间。

/** 树 → 按深度优先排好的行，带缩进层级 */
export function flatten(roots, depth = 0, out = []) {
  for (const n of roots) {
    out.push({ ...n, depth });
    flatten(n.children || [], depth + 1, out);
  }
  return out;
}

/** 整棵树的时间范围；没结束的 span 算到 now（还在跑 / 崩溃后没收尾的也画得出来） */
export function bounds(rows, now = Date.now()) {
  const start = Math.min(...rows.map((r) => Date.parse(r.started_at)));
  const end = Math.max(...rows.map((r) => (r.ended_at ? Date.parse(r.ended_at) : now)));
  return { start, end };
}

/** 瀑布条的位置：相对整棵树的百分比。太短的 span 给个最小宽度，不然看不见 */
export function bar(row, b, now = Date.now()) {
  const s = Date.parse(row.started_at);
  const e = row.ended_at ? Date.parse(row.ended_at) : now;
  const total = Math.max(b.end - b.start, 1);
  return { left: ((s - b.start) / total) * 100, width: Math.max(((e - s) / total) * 100, 0.5) };
}

/** 行尾的一句摘要：llm 看 token 和停止原因，run 看停止原因，工具看报错 */
export function summary(row) {
  const a = row.attrs || {};
  if (row.kind === "llm") {
    const u = a.usage || {};
    const tok = a.usage ? `入 ${u.input_tokens ?? 0} / 缓存 ${u.cache_read_input_tokens ?? 0} / 出 ${u.output_tokens ?? 0}` : "";
    return [tok, a.stop_reason].filter(Boolean).join(" · ");
  }
  if (row.kind === "run") return a.stop_reason || "";
  return row.error ? row.error.slice(0, 60) : "";
}
