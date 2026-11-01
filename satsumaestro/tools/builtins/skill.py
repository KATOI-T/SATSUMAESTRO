"""skill 組み込みツール。"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.skill.resolver import SkillLookupError, list_skills, resolve_skill
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema


class Skill(Tool):
    """ローカルスキル（手順書）の読み込み。"""

    class Input(BaseModel):
        skill: str = Field(description="Name of the local skill to load (case-insensitive).")
        args: str = Field(
            default="",
            description="Optional free-form arguments passed along with the skill instructions.",
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="skill",
        description=(
            "Read a local skill by name and return its instructions. A skill is a procedure "
            "document kept in the project or by the user; once it is loaded, carry out the "
            "current task the way it describes."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
    )
    state_dir: ClassVar[bool] = True

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, Skill.Input)
        root = ctx.workspace_root
        try:
            meta = resolve_skill(root, inp.skill, state_dir=self.state_dir)
        except SkillLookupError as e:
            known = (
                ", ".join(m.name for m in list_skills(root, state_dir=self.state_dir)) or "(none)"
            )
            return ToolResult(output=f"{e} (known skills: {known})", is_error=True)
        try:
            body = meta.path.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult(output=f"failed to read skill {meta.name}: {e}", is_error=True)
        header = f"# Skill: {meta.name}\n"
        if inp.args:
            header += f"(args: {inp.args})\n"
        return ToolResult(output=header + "\n" + body)


class UntrustedSkill(Skill):
    """信頼していないディレクトリ用の skill。"""

    state_dir: ClassVar[bool] = False
