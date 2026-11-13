"""M9：真实沙箱后端。包 e2b 的 AsyncSandbox，连本机 Cube（兼容 E2B 接口）。

- SDK 钉 e2b==2.50.0 / e2b-code-interpreter==2.10.0：≥2.51 会请求 v2 接口，Cube v0.7.3 返回 405；
  code-interpreter <2.10 不认 E2B_SANDBOX_URL。连接参数 E2B_API_URL / E2B_SANDBOX_URL / E2B_API_KEY 由 SDK 自己读环境变量。
- 下面的行为都是 2026-11-02 在 Cube 上实测的（scratchpad probe_m9.py / probe_shell.py）：
  · run_code 超时后内核里的代码还在跑，同一上下文的下一次调用会被堵住 → 池子负责重启内核（restart 约 2 秒）；
  · 重启不存在的上下文报 NotFoundException → 这里转成 False；
  · 前台 shell 命令被取消后进程还在跑 → 用 background=True 拿句柄，取消时自己 kill；
  · 读不存在的文件抛 e2b 自己的 FileNotFoundException（不是内置的）→ 转成内置的，工具层当预期内错误；
  · files.write 会自动建上级目录。
"""

import asyncio
import base64
import contextlib
import os
import posixpath
import time

from e2b import CommandExitException, SandboxQuery
from e2b.exceptions import FileNotFoundException, NotFoundException, SandboxException, SandboxNotFoundException
from e2b_code_interpreter import AsyncSandbox
from e2b_code_interpreter.models import Context

from app import sandbox
from app.sandbox import Output

TEMPLATE = os.getenv("CUBE_TEMPLATE_ID", "tpl-99f0a28ec899400fb3e132a3")
APP = "oasis"   # metadata 标签：启动对账只看带这个标签的沙箱，不碰别人建的
CONTEXT_READY_SECONDS = 60
HOME = "/workspace"  # 实测代码上下文的默认 cwd 是 /workspace；shell 也放这里，相对路径两边一致


class E2BBackend:
    def __init__(self):
        self._boxes: dict[str, AsyncSandbox] = {}  # 连接缓存；沙箱的权威记录在数据库

    async def _box(self, sid: str) -> AsyncSandbox:
        if sid not in self._boxes:
            self._boxes[sid] = await AsyncSandbox.connect(sid)
        return self._boxes[sid]

    async def create(self, user_id: str) -> str:
        box = await AsyncSandbox.create(template=TEMPLATE, timeout=sandbox.TTL_SECONDS,
                                        metadata={"app": APP, "user_id": user_id},
                                        allow_internet_access=False)  # 断网：risk=LOW 的前提之一
        self._boxes[box.sandbox_id] = box
        return box.sandbox_id

    async def alive(self, sid: str) -> bool:
        try:
            await AsyncSandbox.get_info(sid)
            return True
        except (SandboxNotFoundException, NotFoundException):  # 其他错误（Cube 连不上）照常抛，算「没开始」
            self._boxes.pop(sid, None)
            return False

    async def kill(self, sid: str) -> None:
        self._boxes.pop(sid, None)
        await AsyncSandbox.kill(sid)  # 不存在时返回 False，不抛

    async def list_ours(self) -> dict[str, str]:
        page, ours = AsyncSandbox.list(query=SandboxQuery(metadata={"app": APP})), {}
        while page.has_next:
            for info in await page.next_items():
                ours[info.sandbox_id] = info.metadata.get("user_id", "")
        return ours

    async def keepalive(self, sid: str, seconds: int) -> None:
        await AsyncSandbox.set_timeout(sid, seconds)

    async def new_context(self, sid: str) -> str:
        # 实测：新沙箱里第一次建上下文要等内核启动，单个约 10 秒；3 个沙箱同时建约 45 秒，期间报 500。
        # 评测里模型重试几次（不到 30 秒）就放弃、改成心算。这一步还在「没开始」阶段，重试安全
        box, deadline = await self._box(sid), time.monotonic() + CONTEXT_READY_SECONDS
        while True:
            try:
                # 不传 cwd：实测 Cube 要求它在 CODE_INTERPRETER_WORKDIR 下，传 /home/user 报 400
                return (await box.create_code_context()).id
            except SandboxException as e:
                if not str(e).startswith("500") or time.monotonic() > deadline:
                    raise
                await asyncio.sleep(2)

    async def restart_context(self, sid: str, context_id: str) -> bool:
        try:
            await (await self._box(sid)).restart_code_context(context_id)
            return True
        except NotFoundException:
            return False

    async def run_code(self, sid: str, context_id: str, code: str, timeout: float) -> Output:
        ex = await (await self._box(sid)).run_code(code, context=Context(context_id, "python", HOME), timeout=timeout)
        err = f"{ex.error.name}: {ex.error.value}\n{ex.error.traceback}" if ex.error else None
        return Output(stdout="".join(ex.logs.stdout), stderr="".join(ex.logs.stderr), result=ex.text, error=err,
                      images=[base64.b64decode(r.png) for r in ex.results if r.png])

    async def run_shell(self, sid: str, command: str, timeout: float) -> Output:
        handle = await (await self._box(sid)).commands.run(command, background=True, cwd=HOME, timeout=timeout)
        try:
            r = await handle.wait()
        except CommandExitException as e:  # 非 0 退出是正常结果，异常本身就带着输出和退出码
            r = e
        except BaseException:  # 取消 / 超时：重启内核杀不到 shell 进程，自己杀
            with contextlib.suppress(Exception):
                await asyncio.shield(handle.kill())
            raise
        return Output(stdout=r.stdout, stderr=r.stderr, exit_code=r.exit_code)

    async def read_file(self, sid: str, path: str) -> str:
        try:
            return await (await self._box(sid)).files.read(posixpath.join(HOME, path))
        except FileNotFoundException as e:
            raise FileNotFoundError(path) from e

    async def write_file(self, sid: str, path: str, data: str | bytes) -> None:
        # 相对路径按 HOME 解析，和 run_code / run_shell 的 cwd 一致（join 遇到绝对路径会原样返回它）
        await (await self._box(sid)).files.write(posixpath.join(HOME, path), data)
