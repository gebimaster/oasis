"""模型注册表（仿 pi-ai 的 Model 目录）：模型名 → 走哪个协议、连哪里、用哪个 key、窗口/价格/能力。

调用方只传模型名（MODEL / JUDGE_MODEL），协议由这里决定，所以被测模型和评委可以走不同协议。
同一个模型可以从两种协议接入，用「名字@协议」区分；不带后缀的沿用改造前的行为（Anthropic 协议）。
"""

import os
from dataclasses import dataclass, field

ANTHROPIC = "anthropic-messages"
OPENAI_CHAT = "openai-chat"


@dataclass(frozen=True)
class Model:
    api: str                         # 协议：决定用哪个 provider 文件翻译
    id: str                          # 发给服务端的真实模型名
    base_url: str | None = None      # None = 读 SDK 自己的环境变量（ANTHROPIC_BASE_URL）
    key_env: tuple[str, ...] = ("ANTHROPIC_API_KEY",)  # 依次找，第一个有值的生效
    context_window: int = 200_000
    max_output: int = 64_000
    reasoning: bool = False          # 会不会出 thinking
    cost: tuple[float, float, float] = (0, 0, 0)  # 每百万 token 美元：输入 / 输出 / 缓存命中
    # 同协议不同厂商的差异开关（pi-ai 叫 compat）
    compat: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        """thinking 块的来源标签：同一个标签才认为是「同一个模型」（pi-ai 比的是 provider+api+model）。"""
        return f"{self.api}:{self.id}"

    def api_key(self) -> str | None:
        return next((v for k in self.key_env if (v := os.getenv(k))), None)


# 价格、窗口取自 pi-ai catalog/providers/deepseek.ts（注明来源：DeepSeek 官方定价页）
_DS = dict(context_window=1_000_000, max_output=384_000, reasoning=True, cost=(0.3, 1.2, 0.006))

MODELS: dict[str, Model] = {
    # 改造前就在用的：DeepSeek 的 Anthropic 兼容端点，地址读 ANTHROPIC_BASE_URL
    "deepseek-flash": Model(ANTHROPIC, "deepseek-flash", **_DS),
    # 同一个模型走 OpenAI Chat 协议；本机 .env 里 DeepSeek 的 key 放在 ANTHROPIC_API_KEY，所以兜底读它
    "deepseek-flash@openai": Model(OPENAI_CHAT, "deepseek-flash", base_url="https://api.deepseek.com",
                                   key_env=("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"), **_DS,
                                   # 同 pi-ai DEEPSEEK compat；实测 OpenAI 端点默认不思考（Anthropic 端点默认思考），显式打开对齐
                                   compat={"reasoning_content_required": True,
                                           "extra_body": {"thinking": {"type": "enabled"}}}),
    "claude-opus-5-5": Model(ANTHROPIC, "claude-opus-5-5", reasoning=True),  # 窗口/价格未核实，先用默认值
}


def resolve(name: str | None = None) -> Model:
    """查不到的名字按 Anthropic 协议原样透传（改造前的行为），不让新增模型名直接报错。"""
    name = name or os.getenv("MODEL", "claude-opus-5-5")
    return MODELS.get(name) or Model(ANTHROPIC, name)
