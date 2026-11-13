// 前端事件归约：纯函数，不碰 DOM，方便 node --test 单测。
// 去重规则和后端 stream_run 一致：seq <= lastSeq 就是重复或过期，原样返回 state（调用方用 === 判断是否生效）。

export const initial = () => ({ lastSeq: 0, live: "", ended: null });

export function apply(state, ev) {
  if (ev.seq <= state.lastSeq) return state;
  const s = { ...state, lastSeq: ev.seq ?? state.lastSeq }; // M1 旧 run 合成的 run_end 没有 seq
  if (ev.type === "text_delta") s.live = state.live + ev.text;   // 追加到正在流式显示的那段
  else if (ev.type === "message_saved") s.live = "";            // 已落库成消息：交给历史按 message_id 显示
  else if (ev.type === "retract") s.live = "";                  // O7：这次模型调用作废（要重试或失败），推过的字不算数
  else if (ev.type === "run_end") s.ended = ev;
  return s;
}

// 断线重连的退避：每次翻倍，最多 10s；收到新事件后调用方把它重置回 500ms
export const nextDelay = (d) => Math.min(d * 2, 10_000);

// O2：游标持久化。刷新后带着上次的 lastSeq 续播，live 也一起存：lastSeq 之前的增量不会再回放，前半截只能从这里来。
// 只认同一个 run：换了 run（或存的东西坏了）就从 0 开始。sessionStorage 按标签页隔离，两个标签页互不覆盖游标。
export const snapshot = (runId, st) => JSON.stringify({ runId, lastSeq: st.lastSeq, live: st.live });

export function restore(saved, runId) {
  try {
    const s = JSON.parse(saved);
    if (s?.runId === runId && Number.isInteger(s.lastSeq) && typeof s.live === "string")
      return { ...initial(), lastSeq: s.lastSeq, live: s.live };
  } catch { /* null 或坏数据：当没存过 */ }
  return initial();
}
