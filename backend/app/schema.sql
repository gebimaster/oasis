-- M1 表结构。原始消息永不删除；M2 会加 run_events 表。

CREATE TABLE IF NOT EXISTS users (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name       text NOT NULL,
    api_token  text NOT NULL UNIQUE,          -- M1 的简易鉴权：请求头带 token 换出 user_id
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid NOT NULL REFERENCES users(id),
    title      text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid NOT NULL REFERENCES conversations(id),
    status          text NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'completed', 'cancelled', 'failed')),
    stop_reason     text,                     -- 细粒度原因：end_turn / max_turns / refusal / api_error / interrupted ...
    error           text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz
);

-- 同一会话同时最多一个活着的 run：由数据库原子保证，并发请求第二个必失败
-- M7：cancelling（停止意图已落库、旧 task 还在收尾写库）也算活着，否则停止后立刻发新消息会和旧 task 同时写
DROP INDEX IF EXISTS one_running_run_per_conversation;
CREATE UNIQUE INDEX IF NOT EXISTS one_active_run_per_conversation
    ON runs (conversation_id) WHERE status IN ('running', 'cancelling');

CREATE TABLE IF NOT EXISTS messages (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- 自增 id 即排序依据
    conversation_id uuid NOT NULL REFERENCES conversations(id),
    run_id          uuid REFERENCES runs(id),
    role            text NOT NULL CHECK (role IN ('user', 'assistant')),
    content         jsonb NOT NULL,           -- Messages API 的 content 数组原样存（含 thinking 块）
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);

-- M2：事件日志。seq 由 runs.last_seq 在同一事务里分配：事件行和号码要么都提交、要么都回滚，不复用、不留洞
ALTER TABLE runs ADD COLUMN IF NOT EXISTS last_seq int NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS run_events (
    run_id     uuid NOT NULL REFERENCES runs(id),
    seq        int  NOT NULL,
    type       text NOT NULL,             -- text_delta / tool_start / tool_end / message_saved / run_end
    data       jsonb NOT NULL,            -- 推给前端的事件本体（不含 seq）
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)             -- 续播查询 seq > after 直接走这个索引
);

-- M3：消息状态。interrupted = 进程崩溃时只生成了一半、由启动恢复从 run_events 拼回来的助手消息
ALTER TABLE messages ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'completed'
    CHECK (status IN ('completed', 'interrupted'));

-- M3：产生这条 assistant 消息的那次请求的 usage（input/output/cache_read/cache_creation），预算估算打底用
ALTER TABLE messages ADD COLUMN IF NOT EXISTS usage jsonb;

-- M3：压缩状态。原文一行不改，只存「摘要 + 从哪条开始保留原文」；revision 做乐观锁（CAS）
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summary text;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS keep_from_id bigint NOT NULL DEFAULT 0;     -- 水位：id >= 它的消息保留原文
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS compacted_at_id bigint NOT NULL DEFAULT 0;  -- 压缩时最大的消息 id：<= 它的 usage 已过期
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS revision int NOT NULL DEFAULT 0;
-- O4：被水位跳过、但要原样保留的消息副本 [{role, content}]（单轮内压缩时的当前提问）。
-- 加载时拼在摘要之后、水位之后的原文之前；每次提交整份覆盖
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS replay_prefix jsonb NOT NULL DEFAULT '[]';

-- M4：本会话用 tool_search select 加载过的延迟工具（按加载顺序）。压缩会把 tool_search 结果摘要掉，靠它把定义补回
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS loaded_tools jsonb NOT NULL DEFAULT '[]';

-- M4：用户权限，一行一条「资源:动作」字符串（如 stock:read）。没有的就是没有：默认拒绝
CREATE TABLE IF NOT EXISTS user_permissions (
    user_id    uuid NOT NULL REFERENCES users(id),
    permission text NOT NULL,
    PRIMARY KEY (user_id, permission)
);

-- M5：确认规则。user_id 为空 = 全局规则；conditions 是 [{field, op, value}]，全部满足才命中；命中多条 deny > ask > allow
CREATE TABLE IF NOT EXISTS tool_rules (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id      uuid REFERENCES users(id),
    tool_pattern text  NOT NULL,              -- fnmatch 通配，如 transfer、mcp__*
    conditions   jsonb NOT NULL DEFAULT '[]',
    effect       text  NOT NULL CHECK (effect IN ('deny', 'ask', 'allow')),
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- M5：审批记录。一批 tool_use（同一条助手消息）里只要有一个要确认，整批每个调用都落一行：
-- 放行的直接带结果，要确认的 pending；续跑时按 position 拼回一条 tool_result 消息
CREATE TABLE IF NOT EXISTS tool_approval (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id   uuid   NOT NULL REFERENCES conversations(id),
    run_id            uuid   NOT NULL REFERENCES runs(id),       -- 产生这批调用的 run
    message_id        bigint NOT NULL REFERENCES messages(id),   -- tool_use 所在的助手消息 = 批次
    position          int    NOT NULL,                           -- 在批次里的顺序
    tool_use_id       text   NOT NULL UNIQUE,
    tool_name         text   NOT NULL,                           -- call_tool 拆包后的真名
    args_json         text   NOT NULL,                           -- 参数原文，只存这一次；执行只读它，不收前端回传
    args_sha256       text   NOT NULL,                           -- 执行前重算比对，防止记录被改
    decision          text   NOT NULL CHECK (decision IN ('allow', 'ask', 'deny')),
    status            text   NOT NULL CHECK (status IN (
                          'pending', 'approved', 'rejected', 'superseded',       -- 人的决定
                          'executing', 'succeeded', 'failed', 'outcome_unknown', -- 执行
                          'denied')),                                            -- 规则拒绝
    result            jsonb,                                     -- {content, is_error}
    resolution        text CHECK (resolution IN ('done', 'not_done')),  -- 结果未知时用户核实的结论
    result_message_id bigint REFERENCES messages(id),            -- tool_result 写进了哪条消息；NULL = 批次未收口
    created_at        timestamptz NOT NULL DEFAULT now(),
    decided_at        timestamptz,
    finished_at       timestamptz
);
CREATE INDEX IF NOT EXISTS tool_approval_open ON tool_approval (conversation_id) WHERE result_message_id IS NULL;

-- M5：演示写工具 transfer 的账本。数行数就知道执行了几次
CREATE TABLE IF NOT EXISTS ledger (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    uuid    NOT NULL REFERENCES users(id),
    to_name    text    NOT NULL,
    amount     numeric NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- M5：用户点停止的那条助手消息记为 stopped（拼给模型时加「已停止」说明）
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_status_check;
ALTER TABLE messages ADD CONSTRAINT messages_status_check CHECK (status IN ('completed', 'interrupted', 'stopped'));

-- M6：知识库文档登记。块在 Chroma，这里管归属和处理状态（上传是异步的，用户选①b）
CREATE TABLE IF NOT EXISTS kb_documents (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),   -- 服务端生成（用户选②b），也是 Chroma 里的 doc_id
    user_id    uuid NOT NULL REFERENCES users(id),
    title      text NOT NULL,
    status     text NOT NULL CHECK (status IN ('processing', 'ready', 'failed')),
    error      text,                                         -- 只记异常类型，不记原文
    stats      jsonb,                                        -- 最近一次 ingest 的 {added, deleted, kept}
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS kb_documents_user ON kb_documents (user_id, created_at);

-- M7：子智能体。子 run 在自己的子会话里跑：不撞「一个会话一个 running」的锁，也不混进父会话的历史
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS parent_conversation_id uuid REFERENCES conversations(id);  -- 非空 = 子会话，不进会话列表
ALTER TABLE runs ADD COLUMN IF NOT EXISTS parent_run_id uuid REFERENCES runs(id);  -- 哪个父 run 派出的；启动对账靠它
ALTER TABLE runs ADD COLUMN IF NOT EXISTS parent_tool_use_id text;                 -- 父 run 里哪一次 spawn_subagents
ALTER TABLE runs ADD COLUMN IF NOT EXISTS title text;                              -- 胶囊上显示的名字
ALTER TABLE runs ADD COLUMN IF NOT EXISTS child_index int;                         -- 第几个子任务（刷新后胶囊按它对上）
ALTER TABLE runs ADD COLUMN IF NOT EXISTS usage jsonb;                             -- 子 run 的 token 合计（耗时 = finished_at - created_at），留给 M8
CREATE INDEX IF NOT EXISTS runs_by_parent ON runs (parent_run_id) WHERE parent_run_id IS NOT NULL;

-- M7 第 5 步：先写库再取消。cancelling = 用户的停止意图已落库，task 还没收尾；进程死在中间，重启也知道是用户停的
ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_status_check;
ALTER TABLE runs ADD CONSTRAINT runs_status_check
    CHECK (status IN ('running', 'cancelling', 'completed', 'cancelled', 'failed'));

-- M7：high 风险工具的执行记录（规则直接放行、没走审批的也记，父和子都记）。
-- 执行前写 executing；被取消、超时、崩溃 → outcome_unknown，不重试，汇报里如实写「结果未知」
CREATE TABLE IF NOT EXISTS tool_executions (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id      uuid  NOT NULL REFERENCES runs(id),
    tool_use_id text  NOT NULL,
    tool_name   text  NOT NULL,
    args        jsonb NOT NULL,
    status      text  NOT NULL CHECK (status IN ('executing', 'succeeded', 'failed', 'outcome_unknown')),
    result      jsonb,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS tool_executions_by_run ON tool_executions (run_id);

-- M8：trace。一个 span = 一段有起止的工作（run / 一次 LLM 调用 / 一次工具调用）。
-- 开始就插入（running），结束再更新：进程死在中间也看得到卡在哪；启动时 running 改 interrupted。
-- 工具的参数和结果不复制进来（用户要能看全文，全文已在 messages），attrs 只存 tool_use_id，查看时再关联
CREATE TABLE IF NOT EXISTS spans (
    id         uuid PRIMARY KEY,                       -- 应用生成：结束时按它更新，不用等 INSERT 返回
    trace_id   uuid NOT NULL,                          -- 根 run 的 id：子 run 的 span 也在父的 trace 里
    parent_id  uuid REFERENCES spans(id),              -- 子 run 挂在派它的 spawn_subagents 工具 span 下（用户选 B）
    run_id     uuid NOT NULL REFERENCES runs(id),      -- 这段工作属于哪个 run（子 run 的 span 填子 run）
    kind       text NOT NULL CHECK (kind IN ('run', 'llm', 'tool')),
    name       text NOT NULL,
    status     text NOT NULL DEFAULT 'running'
               CHECK (status IN ('running', 'ok', 'error', 'cancelled', 'interrupted')),
    attrs      jsonb NOT NULL DEFAULT '{}',            -- llm：turn / usage / stop_reason；tool：tool_use_id；run：stop_reason
    error      text,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),  -- now() 是事务开始时间，这里要真实时刻
    ended_at   timestamptz
);
CREATE INDEX IF NOT EXISTS spans_by_trace ON spans (trace_id, started_at);

-- O1：run 归属。owner_id = 哪个进程在跑（进程每次启动随机生成）；lease_until = 租约到期时刻，一律用数据库 now()。
-- 属主活着就续租；过期的由任意实例的周期扫描接管。可空、无默认值：只改元数据，不重写表
ALTER TABLE runs ADD COLUMN IF NOT EXISTS owner_id uuid;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS lease_until timestamptz;
CREATE INDEX IF NOT EXISTS runs_active_lease ON runs (lease_until) WHERE status IN ('running', 'cancelling');

-- M9：沙箱。一个用户一个沙箱（user_id 主键，并发首调靠它兜底），每个会话在里面一个代码上下文。
-- needs_reset 是「执行前先写」的标记：执行前置 true、正常结束置 false；取消 / 超时 / 崩溃时留着，下次用之前先重启内核
CREATE TABLE IF NOT EXISTS sandboxes (
    user_id      uuid PRIMARY KEY REFERENCES users(id),
    sandbox_id   text NOT NULL UNIQUE,
    created_at   timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sandbox_contexts (
    conversation_id uuid PRIMARY KEY REFERENCES conversations(id),
    sandbox_id      text NOT NULL,  -- 不加外键：沙箱回收 / 重建后这行还在，对不上就知道要告诉模型「变量和文件都没了」
    context_id      text NOT NULL,
    needs_reset     boolean NOT NULL DEFAULT false
);

-- O9 第一步：大工具结果的全文。消息里只存头尾预览和 ref，read_tool_output 按 ref 回取（只能读本人的）。
-- id 由会话、tool_use_id 和内容哈希组成：同一结果重复写入是幂等的
CREATE TABLE IF NOT EXISTS tool_outputs (
    id              text PRIMARY KEY,
    user_id         uuid NOT NULL REFERENCES users(id),
    conversation_id uuid REFERENCES conversations(id),
    tool_name       text NOT NULL,
    content         text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);
-- O9 第二步：清旧工具结果的水位。id <= cleared_until_id 的消息里，可清的工具结果在拼请求时换成固定占位符（库里原文不动）；
-- cleared_at_id 是清理那一刻最大的消息 id：<= 它的 usage 是清理前那份大请求的，不能拿来打底估算（同 compacted_at_id）
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS cleared_until_id bigint NOT NULL DEFAULT 0;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS cleared_at_id bigint NOT NULL DEFAULT 0;

-- O10 第二步：transfer 的幂等键（= tool_use_id）。同一用户同一个键只会有一行：重放、续跑重复执行都只扣一次，
-- 结果未知时按键一查就知道转没转（把 transfer 当成外部 API：只能调它对外的「按键查询」）
ALTER TABLE ledger ADD COLUMN IF NOT EXISTS idem_key text;
CREATE UNIQUE INDEX IF NOT EXISTS ledger_idem ON ledger (user_id, idem_key) WHERE idem_key IS NOT NULL;
-- 结果未知的结论是谁给的：user（人核实）/ reconcile（系统按幂等键查到）
ALTER TABLE tool_approval ADD COLUMN IF NOT EXISTS resolved_by text;
-- O10 复核①：这一行是不是由带幂等键的代码执行的（claim 时置 true）。O10 之前执行的那笔在账本里没有键，
-- 按键查不到不代表没转，对账对它们一律不下结论
ALTER TABLE tool_approval ADD COLUMN IF NOT EXISTS keyed boolean NOT NULL DEFAULT false;
-- O10 复核③：排在「要确认」的调用后面、本来放行的调用先不执行（deferred），批准后续跑按 position 顺序执行
ALTER TABLE tool_approval DROP CONSTRAINT IF EXISTS tool_approval_status_check;
ALTER TABLE tool_approval ADD CONSTRAINT tool_approval_status_check CHECK (status IN (
    'pending', 'approved', 'rejected', 'superseded', 'executing', 'succeeded', 'failed', 'outcome_unknown',
    'denied', 'deferred'));
