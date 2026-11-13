// 运行：node --test backend/static/stream.test.mjs
import assert from "node:assert/strict";
import test from "node:test";
import { apply, initial, nextDelay, restore, snapshot } from "./stream.js";

const delta = (seq, text) => ({ type: "text_delta", seq, text });
const fold = (events, st = initial()) => events.reduce(apply, st);

test("同一个事件收到两次：第二次原样返回，字不重复", () => {
  const st = fold([delta(1, "北京")]);
  assert.equal(apply(st, delta(1, "北京")), st);
  assert.equal(st.live, "北京");
});

test("断线重连时回放和实时有重叠：每个 seq 只生效一次", () => {
  const before = fold([delta(1, "a"), delta(2, "b"), delta(3, "c")]);
  // 重连带 Last-Event-ID=3，但服务端订阅与查库之间推过的事件可能再出现一次
  const after = fold([delta(2, "b"), delta(3, "c"), delta(4, "d"), delta(4, "d"), delta(5, "e")], before);
  assert.equal(after.live, "abcde");
  assert.equal(after.lastSeq, 5);
});

test("刷新后从 0 回放：已落库的那段被 message_saved 清掉，不会和历史重复显示", () => {
  const st = fold([
    delta(1, "我查一下"),
    { type: "message_saved", seq: 2, message_id: 10, role: "assistant" },
    { type: "tool_start", seq: 3, name: "get_weather" },
    delta(4, "北京晴"),
  ]);
  assert.equal(st.live, "北京晴");
  assert.equal(st.ended, null);
});

test("run_end 结束；M1 旧 run 合成的 run_end 没有 seq 也能结束", () => {
  assert.equal(fold([delta(1, "x"), { type: "run_end", seq: 2, status: "completed" }]).ended.status, "completed");
  const legacy = apply(initial(), { type: "run_end", status: "failed", stop_reason: "interrupted" });
  assert.equal(legacy.ended.stop_reason, "interrupted");
});

test("退避翻倍并封顶 10s", () => {
  assert.deepEqual([500, 1000, 8000, 10_000].map(nextDelay), [1000, 2000, 10_000, 10_000]);
});

test("O2 刷新：同一个 run 恢复游标和半截文字，接着续播不重不漏", () => {
  const st = fold([delta(1, "北"), delta(2, "京")]);
  const back = restore(snapshot("r1", st), "r1");
  assert.equal(back.lastSeq, 2);
  assert.equal(back.live, "北京");                     // live 区刷新后立刻有前半截，不闪
  assert.equal(fold([delta(2, "京"), delta(3, "晴")], back).live, "北京晴");
});

test("O2 刷新：换了 run、没存过、存坏了，都从 0 开始", () => {
  const saved = snapshot("r1", fold([delta(5, "x")]));
  assert.deepEqual(restore(saved, "r2"), initial());
  assert.deepEqual(restore(null, "r1"), initial());
  assert.deepEqual(restore("{oops", "r1"), initial());
  assert.deepEqual(restore(JSON.stringify({ runId: "r1", lastSeq: "5", live: "x" }), "r1"), initial());
});

test("O7 retract：作废的那段字清掉，重试后的字只显示一份", () => {
  const st = fold([delta(1, "我查"), { type: "retract", seq: 2 }, delta(3, "我查一下")]);
  assert.equal(st.live, "我查一下");
});

test("O7 retract 也走 seq：刷新后带着游标续播，快照里的半截字随 retract 一起清掉", () => {
  const before = fold([delta(1, "我查")]);
  const saved = snapshot("r1", before);                       // 刷新前存的快照里有「我查」
  const after = fold([{ type: "retract", seq: 2 }, delta(3, "好")], restore(saved, "r1"));
  assert.equal(after.live, "好");
  assert.equal(apply(after, { type: "retract", seq: 2 }), after);  // 重放的旧 retract 不会把新字清掉
});
