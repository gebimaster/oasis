"""O6 第二步的测试辅助：像 tools.execute 一样执行，但把错误码也带回来（错误码不进 tool_result 块）。"""

from app import tools


async def execute_with_code(name: str, args) -> tuple[str, bool, str | None]:
    call = tools.prepare({"id": "", "name": name, "input": args}, check_rules=False)
    content, is_error = await tools.run(call)
    return content, is_error, call.error_code
