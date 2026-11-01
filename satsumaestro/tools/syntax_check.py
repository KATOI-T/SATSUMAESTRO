"""編集直後の構文検査と、diff 形式の本文の検知。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

FLAG_REJECT = "syntax-reject"
FLAG_WARNING = "syntax-warning"

_LANGUAGES: dict[str, str] = {
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".py": "python",
    ".json": "json",
}
NODE_TIMEOUT_S = 10.0
_PAIR_LIMIT = 40
_PAIR_RATIO = 0.6

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.M)
_FILE_HEADER_RE = re.compile(r"^(?:--- (?:a/|/dev/null)|\+\+\+ (?:b/|/dev/null))", re.M)
_CONFLICT_RE = re.compile(r"^(?:<<<<<<< |>>>>>>> )", re.M)
_NODE_LOCATION_RE = re.compile(r"^(.*):(\d+)\s*$")
_NODE_MESSAGE_RE = re.compile(r"^(\w*Error(?::.*)?)$", re.M)
_ESM_ONLY_RE = re.compile(
    r"Cannot use import statement|Unexpected token 'export'|import\.meta|"
    r"await is only valid|top level bodies of modules"
)


@dataclass(frozen=True)
class SyntaxVerdict:
    """1 回の検査結果。"""

    ok: bool
    line: int | None = None
    message: str = ""

    def describe(self) -> str:
        where = f"line {self.line}" if self.line is not None else "unknown line"
        return f"{where}: {self.message}" if self.message else where


@dataclass(frozen=True)
class EditCheck:
    """編集前後の検査結果。"""

    before: SyntaxVerdict | None
    after: SyntaxVerdict | None

    @property
    def after_broken(self) -> bool:
        return self.after is not None and not self.after.ok

    @property
    def before_ok(self) -> bool:
        return self.before is not None and self.before.ok

    @property
    def before_known_broken(self) -> bool:
        return self.before is not None and not self.before.ok


def language_of(path: Path) -> str | None:
    """検査できる言語名（拡張子から。対象外なら None）。"""
    return _LANGUAGES.get(path.suffix.lower())


def node_path() -> str | None:
    """``node`` の場所（無ければ None。JavaScript は検査しない）。"""
    return shutil.which("node")


def check_syntax(path: Path, text: str) -> SyntaxVerdict | None:
    """``text`` を ``path`` の言語として構文検査する。"""
    lang = language_of(path)
    if lang == "javascript":
        return _check_javascript(path, text)
    if lang == "python":
        return _check_python(path, text)
    if lang == "json":
        return _check_json(text)
    return None


def before_text(path: Path) -> str | None:
    """検査用に編集前の本文を読む（無ければ None＝新規ファイル）。"""
    return path.read_text(encoding="utf-8") if path.is_file() else None


def check_edit(path: Path, before_text: str | None, after_text: str) -> EditCheck:
    """編集前後を検査する。"""
    after = check_syntax(path, after_text)
    if after is None or after.ok or before_text is None:
        return EditCheck(before=None, after=after)
    return EditCheck(before=check_syntax(path, before_text), after=after)


def _check_python(path: Path, text: str) -> SyntaxVerdict | None:
    try:
        compile(text, str(path), "exec")
    except SyntaxError as e:
        return SyntaxVerdict(ok=False, line=e.lineno, message=f"SyntaxError: {e.msg}")
    except ValueError as e:
        return SyntaxVerdict(ok=False, line=None, message=f"ValueError: {e}")
    except Exception:
        return None
    return SyntaxVerdict(ok=True)


def _check_json(text: str) -> SyntaxVerdict:
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return SyntaxVerdict(ok=False, line=e.lineno, message=f"JSON: {e.msg}")
    return SyntaxVerdict(ok=True)


def _check_javascript(path: Path, text: str) -> SyntaxVerdict | None:
    """``node --check`` を一時ファイルで回す（元のパスは触らない）。"""
    node = node_path()
    if node is None:
        return None
    suffix = path.suffix.lower()
    if suffix != ".js":
        return _node_check(node, text, suffix)
    as_cjs = _node_check(node, text, ".cjs")
    if as_cjs is None or as_cjs.ok:
        return as_cjs
    as_esm = _node_check(node, text, ".mjs")
    if as_esm is None or as_esm.ok:
        return as_esm
    return as_esm if _ESM_ONLY_RE.search(as_cjs.message) else as_cjs


def _node_check(node: str, text: str, suffix: str) -> SyntaxVerdict | None:
    """``text`` を拡張子 ``suffix`` の一時ファイルに書いて ``node --check`` する。"""
    try:
        with tempfile.TemporaryDirectory(prefix="satsuma-syntax-") as tmp:
            target = Path(tmp) / f"check{suffix}"
            target.write_text(text, encoding="utf-8")
            proc = subprocess.run(
                [node, "--check", str(target)],
                capture_output=True,
                text=True,
                timeout=NODE_TIMEOUT_S,
            )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return SyntaxVerdict(ok=True)
    message = _node_message(proc.stderr)
    if message is None:
        return None
    return SyntaxVerdict(ok=False, line=_node_line(proc.stderr), message=message)


def _node_line(stderr: str) -> int | None:
    for line in stderr.splitlines():
        m = _NODE_LOCATION_RE.match(line)
        if m:
            return int(m.group(2))
    return None


def _node_message(stderr: str) -> str | None:
    m = _NODE_MESSAGE_RE.search(stderr)
    return m.group(1).strip()[:200] if m else None


def looks_like_patch(text: str) -> str | None:
    """``text`` が diff / conflict marker に見える理由（見えなければ None）。"""
    if _CONFLICT_RE.search(text):
        return "it contains merge conflict markers (<<<<<<< / >>>>>>>)"
    if _HUNK_RE.search(text) or _FILE_HEADER_RE.search(text):
        return "it has unified diff headers (@@ … @@ or --- a/ … +++ b/)"
    minus: list[str] = []
    plus: list[str] = []
    for line in text.splitlines():
        if line.startswith("-") and not line.startswith("--"):
            body = line[1:].strip()
            if body:
                minus.append(body)
        elif line.startswith("+") and not line.startswith("++"):
            body = line[1:].strip()
            if body:
                plus.append(body)
    for m in minus[:_PAIR_LIMIT]:
        for p in plus[:_PAIR_LIMIT]:
            if SequenceMatcher(None, m, p).ratio() >= _PAIR_RATIO:
                return "it has paired '-' and '+' lines like a unified diff"
    return None


def patch_rejection(path: Path, new: str, *, old: str = "") -> str | None:
    """``new`` を diff とみなして拒否する理由。"""
    if language_of(path) is None:
        return None
    reason = looks_like_patch(new)
    if reason is None or (old and looks_like_patch(old) is not None):
        return None
    return reason


def reject_patch_message(path: str, reason: str) -> str:
    return (
        f"edit rejected: `new` looks like a patch ({reason}). Pass the replacement text itself, "
        "i.e. the lines exactly as they should appear in the file, not a diff. "
        f"{path} was left unchanged."
    )


def reject_syntax_message(path: str, verdict: SyntaxVerdict) -> str:
    return (
        f"edit rejected: it would break the syntax of {path} ({verdict.describe()}). "
        "The file was left unchanged. Fix the replacement text (check brackets, quotes and "
        "comment markers; do not paste a diff) and retry."
    )


def break_warning(path: str, verdict: SyntaxVerdict) -> str:
    """通っていたファイルを壊したが適用したときに結果へ添える。"""
    return (
        f"\nwarning: {path} no longer parses ({verdict.describe()}). It parsed before this "
        "change. Fix it now with edit_file before doing anything else."
    )


def write_warning(path: str, verdict: SyntaxVerdict) -> str:
    """write_file の結果に添える（新規・既存を問わず本文が通らないとき）。"""
    return (
        f"\nwarning: {path} does not parse ({verdict.describe()}). If this is the first part of "
        "a longer file, continue with append_file and make sure the last part completes it; "
        "otherwise fix it now with edit_file."
    )


def append_note(path: str, verdict: SyntaxVerdict) -> str:
    """append_file の結果に添える（途中状態は正当なので短い注記だけ）。"""
    return (
        f"\nnote: {path} does not parse yet ({verdict.describe()}); make sure the final part "
        "completes it."
    )


def still_broken_note(path: str, verdict: SyntaxVerdict) -> str:
    """編集前から通っていないファイルへの edit_file の結果に添える（止めない）。"""
    return f"\nnote: {path} still does not parse ({verdict.describe()})."
