"""bash 読み取り自動許可の判定。"""

from __future__ import annotations

import fnmatch
import glob
import os
import re
import shlex
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from satsumaestro.config.models import PermissionConfig
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import _glob_to_regex
from satsumaestro.util.errors import PathEscapeError

_DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "*.p12",
    "*.pfx",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "credentials*",
    "secrets.*",
    ".ssh/**",
    ".aws/**",
    ".kube/**",
)


class _GlobSet:
    """glob の集まりの照合（basename と workspace 相対パスの両方で評価）。"""

    def __init__(self, patterns: Sequence[str]) -> None:
        self.patterns: tuple[str, ...] = tuple(dict.fromkeys(p for p in patterns if p))
        self._basename = [p.lower() for p in self.patterns if "/" not in p]
        self._paths = [
            re.compile(r"(?:.*/)?" + _glob_to_regex(p.strip("/")), re.IGNORECASE)
            for p in self.patterns
            if "/" in p
        ]

    def matches(self, rel: str) -> bool:
        base = rel.rsplit("/", 1)[-1].lower()
        if any(fnmatch.fnmatchcase(base, p) for p in self._basename):
            return True
        return any(r.fullmatch(rel) for r in self._paths)


class SecretMatcher:
    """秘密ファイルパターンの照合。"""

    def __init__(self, patterns: Sequence[str], exclude: Sequence[str] = ()) -> None:
        self._secret = _GlobSet(patterns)
        self._exclude = _GlobSet(exclude)
        self.patterns = self._secret.patterns
        self.exclude = self._exclude.patterns

    @classmethod
    def from_config(cls, cfg: PermissionConfig) -> SecretMatcher:
        """既定パターン + ``secret_patterns`` − ``secret_patterns_remove``。"""
        merged = [*_DEFAULT_SECRET_PATTERNS, *cfg.secret_patterns]
        remove = set(cfg.secret_patterns_remove)
        exclude = [r for r in cfg.secret_patterns_remove if r not in merged]
        return cls([p for p in merged if p not in remove], exclude)

    def matches(self, rel_path: str) -> bool:
        rel = rel_path.replace(os.sep, "/").strip("/")
        if not rel or self._exclude.matches(rel):
            return False
        return self._secret.matches(rel)


def secret_path_hit(path: str, *, jail: WorkspaceJail, secrets: SecretMatcher) -> str | None:
    """ツールの ``path`` 引数が秘密ファイルを指すなら、その workspace 相対パス。"""
    try:
        resolved = jail.resolve(path)
    except PathEscapeError:
        return None
    rel = "" if resolved == jail.root else resolved.relative_to(jail.root).as_posix()
    if secrets.matches(rel):
        return rel
    if secrets.matches(path):
        return path
    return None


@dataclass(frozen=True)
class AutoAllowVerdict:
    """``allow`` = 降格可 / ``reject`` = ask のまま / ``secret`` = 秘密パス該当。"""

    kind: Literal["allow", "reject", "secret"]
    reason: str = ""
    paths: tuple[str, ...] = ()


def _reject(reason: str) -> AutoAllowVerdict:
    return AutoAllowVerdict(kind="reject", reason=reason)


_NO_PATH_ARG_COMMANDS = frozenset(
    {
        "pwd",
        "echo",
        "printf",
        "cal",
        "expr",
        "seq",
        "true",
        "false",
        "which",
        "whereis",
        "whatis",
        "type",
        "uname",
        "hostname",
        "whoami",
        "id",
        "groups",
        "uptime",
        "nproc",
        "arch",
        "sw_vers",
        "basename",
        "dirname",
        "date",
    }
)
_PATH_ARG_COMMANDS = frozenset(
    {
        "cat",
        "head",
        "tail",
        "nl",
        "tac",
        "wc",
        "sort",
        "uniq",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "diff",
        "cmp",
        "comm",
        "md5sum",
        "sha1sum",
        "sha256sum",
        "shasum",
        "xxd",
        "od",
        "hexdump",
        "strings",
        "jq",
        "tr",
        "cut",
        "paste",
        "column",
        "bc",
        "ls",
        "tree",
        "file",
        "stat",
        "du",
        "df",
        "readlink",
        "realpath",
        "find",
        "test",
        "[",
        "cd",
        "git",
    }
)
_AUTO_ALLOW_COMMANDS = _NO_PATH_ARG_COMMANDS | _PATH_ARG_COMMANDS

_SEPARATORS = frozenset({";", "&&", "||", "|"})
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_BRACE_EXPANSION = re.compile(r"\{[^{}]*(?:,|\.\.)[^{}]*\}")
_GLOB_CHARS = frozenset("*?[")


def _forbidden_option(tok: str, prefixes: tuple[str, ...], short: frozenset[str]) -> bool:
    """``tok`` が禁止オプションに該当するか（省略形・付加値を含めて保守側に倒す）。"""
    if tok.startswith("--"):
        name = tok[2:].split("=", 1)[0]
        if not name:
            return False
        for p in prefixes:
            if not p.startswith("--"):
                continue
            pname = p[2:].split("=", 1)[0]
            if pname.startswith(name) or name.startswith(pname):
                return True
        return False
    if tok.startswith("-") and len(tok) > 1:
        if short and set(tok[1:]) & short:
            return True
        return any(tok.startswith(p) for p in prefixes if not p.startswith("--"))
    return False


@dataclass(frozen=True)
class CommandRule:
    """1 つでも該当すれば不許可。"""

    forbidden_tokens: frozenset[str] = frozenset()
    forbidden_prefixes: tuple[str, ...] = ()
    forbidden_short: frozenset[str] = frozenset()
    value_opts: frozenset[str] = frozenset()
    max_positionals: int | None = None
    extra: Callable[[list[str], Path], str | None] | None = None

    def violation(self, args: list[str], cwd: Path) -> str | None:
        for tok in args:
            if tok in self.forbidden_tokens:
                return f"option {tok}"
            if _forbidden_option(tok, self.forbidden_prefixes, self.forbidden_short):
                return f"option {tok}"
        if self.max_positionals is not None:
            count = len(_positionals(args, self.value_opts))
            if count > self.max_positionals:
                return f"{count} positional arguments (max {self.max_positionals})"
        if self.extra is not None:
            return self.extra(args, cwd)
        return None


def _positionals(args: list[str], value_opts: frozenset[str] = frozenset()) -> list[str]:
    """``-`` で始まらないトークン（``--`` 以降は全て）。"""
    out: list[str] = []
    skip = False
    after_dashdash = False
    for tok in args:
        if skip:
            skip = False
            continue
        if after_dashdash:
            out.append(tok)
            continue
        if tok == "--":
            after_dashdash = True
            continue
        if tok.startswith("-") and tok != "-":
            if tok in value_opts:
                skip = True
            continue
        out.append(tok)
    return out


def _date_rule(args: list[str], cwd: Path) -> str | None:
    for tok in _positionals(args):
        if not tok.startswith("+"):
            return f"date positional {tok} (may set the clock)"
    return None


def _jq_rule(args: list[str], cwd: Path) -> str | None:
    for tok in args:
        if re.search(r"\benv\b", tok):
            return "jq env builtin dumps the environment"
    return None


_RG_VALUE_OPTS = frozenset(
    {
        "-e",
        "--regexp",
        "-f",
        "--file",
        "-g",
        "--glob",
        "--iglob",
        "-t",
        "--type",
        "-T",
        "--type-not",
        "--type-add",
        "-A",
        "--after-context",
        "-B",
        "--before-context",
        "-C",
        "--context",
        "-m",
        "--max-count",
        "-M",
        "--max-columns",
        "-j",
        "--threads",
        "-E",
        "--encoding",
        "-r",
        "--replace",
        "--color",
        "--colors",
        "--sort",
        "--sortr",
        "--max-depth",
        "--max-filesize",
        "--ignore-file",
        "--context-separator",
        "--path-separator",
    }
)


def _rg_rule(args: list[str], cwd: Path) -> str | None:
    positionals = _positionals(args, _RG_VALUE_OPTS)
    if len(positionals) < 2:
        return "rg requires explicit file arguments (directory search is recursive)"
    for tok in positionals[1:]:
        target = Path(tok) if Path(tok).is_absolute() else cwd / tok
        if not target.is_file():
            return f"rg target is not a regular file: {tok}"
    return None


_RULES: dict[str, CommandRule] = {
    "find": CommandRule(
        forbidden_tokens=frozenset(
            {
                "-exec",
                "-execdir",
                "-ok",
                "-okdir",
                "-delete",
                "-fprint",
                "-fprintf",
                "-fprint0",
                "-fls",
            }
        )
    ),
    "grep": CommandRule(
        forbidden_prefixes=("--recursive", "--dereference-recursive", "--directories"),
        forbidden_short=frozenset({"r", "R", "d"}),
    ),
    "diff": CommandRule(forbidden_prefixes=("--recursive",), forbidden_short=frozenset({"r"})),
    "rg": CommandRule(
        forbidden_prefixes=(
            "--pre",
            "--hostname-bin",
            "--follow",
            "--hidden",
            "--unrestricted",
            "--no-ignore",
        ),
        forbidden_short=frozenset({"L", "u", "."}),
        extra=_rg_rule,
    ),
    "sort": CommandRule(
        forbidden_prefixes=("--output", "--temporary-directory", "--compress-program"),
        forbidden_short=frozenset({"o", "T"}),
    ),
    "uniq": CommandRule(value_opts=frozenset({"-f", "-s", "-w"}), max_positionals=1),
    "xxd": CommandRule(
        value_opts=frozenset({"-c", "-l", "-s", "-g", "-o", "-cols", "-len", "-seek"}),
        max_positionals=1,
    ),
    "tree": CommandRule(forbidden_short=frozenset({"o"})),
    "file": CommandRule(forbidden_prefixes=("--compile",), forbidden_short=frozenset({"C"})),
    "date": CommandRule(
        forbidden_prefixes=("--set", "--file"),
        forbidden_short=frozenset({"s", "f"}),
        extra=_date_rule,
    ),
    "hostname": CommandRule(
        forbidden_prefixes=("--file",), forbidden_short=frozenset({"F"}), max_positionals=0
    ),
    "jq": CommandRule(extra=_jq_rule),
}
_RULES["egrep"] = _RULES["grep"]
_RULES["fgrep"] = _RULES["grep"]

_GIT_UNCONDITIONAL = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "ls-files",
        "ls-tree",
        "cat-file",
        "rev-parse",
        "rev-list",
        "describe",
        "shortlog",
        "blame",
        "grep",
        "check-ignore",
        "name-rev",
        "count-objects",
    }
)
_GIT_GLOBAL_NOARG = frozenset({"--no-pager", "--no-optional-locks"})
_GIT_FORBIDDEN_EXACT = frozenset({"-o"})
_GIT_FORBIDDEN_PREFIXES = (
    "--output",
    "--ext-diff",
    "--textconv",
    "--exec",
    "--upload-pack",
    "--receive-pack",
)
_GIT_GREP_FORBIDDEN_PREFIXES = ("-O", "--open-files-in-pager")
_GIT_GREP_SHORT = frozenset({"O"})
_GIT_BRANCH_FLAGS = frozenset(
    {"-a", "-r", "-l", "--list", "-v", "-vv", "--all", "--remotes", "--verbose", "--show-current"}
)
_GIT_BRANCH_PREFIXES = (
    "--contains=",
    "--merged=",
    "--no-merged=",
    "--points-at=",
    "--sort=",
    "--format=",
)
_GIT_TAG_PREFIXES = ("-n", "--sort=", "--contains=", "--merged=", "--points-at=", "--format=")


def _git_branch_ok(rest: list[str]) -> bool:
    return all(t in _GIT_BRANCH_FLAGS or t.startswith(_GIT_BRANCH_PREFIXES) for t in rest)


def _git_tag_ok(rest: list[str]) -> bool:
    if not rest:
        return True
    listing = any(t in ("-l", "--list") for t in rest)
    for t in rest:
        if t in ("-l", "--list") or t.startswith(_GIT_TAG_PREFIXES):
            continue
        if t.startswith("-") or not listing:
            return False
    return True


def _git_remote_ok(rest: list[str]) -> bool:
    return all(t in ("-v", "--verbose") for t in rest)


def _git_stash_ok(rest: list[str]) -> bool:
    return bool(rest) and rest[0] in ("list", "show")


def _git_worktree_ok(rest: list[str]) -> bool:
    return bool(rest) and rest[0] == "list"


def _git_reflog_ok(rest: list[str]) -> bool:
    return not any(t in ("expire", "delete") for t in rest)


_GIT_CONDITIONAL: dict[str, Callable[[list[str]], bool]] = {
    "branch": _git_branch_ok,
    "tag": _git_tag_ok,
    "remote": _git_remote_ok,
    "stash": _git_stash_ok,
    "worktree": _git_worktree_ok,
    "reflog": _git_reflog_ok,
}
_GIT_INDEX_PREFIX = re.compile(r":(?:\d+:)?")


def _git_path_forms(t: str) -> list[str]:
    """git の位置引数から検査すべきパス形を返す。"""
    forms = [t]
    if t.startswith(":"):
        rest = _GIT_INDEX_PREFIX.sub("", t, count=1)
        if rest.startswith("(") and ")" in rest:
            rest = rest.split(")", 1)[1]
        if rest:
            forms.append(rest)
    elif ":" in t:
        forms.append(t.split(":", 1)[1])
    return forms


def _git_candidates(args: list[str]) -> tuple[list[str], str | None]:
    """git 引数を検査し、(パス候補, 不許可理由) を返す。"""
    paths: list[str] = []
    i = 0
    while i < len(args) and args[i].startswith("-"):
        tok = args[i]
        if tok == "-C":
            if i + 1 >= len(args):
                return paths, "git -C without a path"
            paths.append(args[i + 1])
            i += 2
            continue
        if tok in _GIT_GLOBAL_NOARG:
            i += 1
            continue
        if tok == "--version":
            return paths, None
        return paths, f"git global option {tok}"
    if i >= len(args):
        return paths, "bare git"
    sub, rest = args[i], args[i + 1 :]
    for t in rest:
        if t in _GIT_FORBIDDEN_EXACT or _forbidden_option(t, _GIT_FORBIDDEN_PREFIXES, frozenset()):
            return paths, f"git option {t}"
        if sub == "grep" and _forbidden_option(t, _GIT_GREP_FORBIDDEN_PREFIXES, _GIT_GREP_SHORT):
            return paths, f"git grep option {t}"
    for t in _option_values(rest):
        paths.extend(_git_path_forms(t))
    if sub in _GIT_UNCONDITIONAL:
        for t in _positionals(rest):
            paths.extend(_git_path_forms(t))
        return paths, None
    check = _GIT_CONDITIONAL.get(sub)
    if check is None:
        return paths, f"git subcommand {sub}"
    if not check(rest):
        return paths, f"git {sub} with non-listing arguments"
    for t in _positionals(rest):
        paths.extend(_git_path_forms(t))
    return paths, None


@dataclass
class _Segment:
    words: list[str] = field(default_factory=list)
    extra_paths: list[str] = field(default_factory=list)
    sep_before: str | None = None
    sep_after: str | None = None


def _is_punct(tok: str) -> bool:
    return bool(tok) and all(c in "();<>|&" for c in tok)


def _tokenize(command: str) -> list[str] | None:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return None


def split_command_segments(command: str) -> list[_Segment] | str:
    """コマンドをセグメントへ分割する。"""
    tokens = _tokenize(command)
    if tokens is None:
        return "unparsable command"
    segments: list[_Segment] = []
    cur = _Segment()
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not _is_punct(tok):
            cur.words.append(tok)
            i += 1
            continue
        if tok in _SEPARATORS:
            if not cur.words:
                return f"empty command before {tok}"
            cur.sep_after = tok
            segments.append(cur)
            cur = _Segment(sep_before=tok)
            i += 1
            continue
        nxt = tokens[i + 1] if i + 1 < n else None
        if tok == "<":
            if nxt is None or _is_punct(nxt):
                return "input redirection without a file"
            cur.extra_paths.append(nxt)
            i += 2
            continue
        if tok == ">&" and nxt is not None and nxt.isdigit():
            _drop_trailing_fd(cur)
            i += 2
            continue
        if tok == ">" and nxt == "/dev/null":
            _drop_trailing_fd(cur)
            i += 2
            continue
        return f"operator {tok}"
    if cur.words:
        segments.append(cur)
    elif cur.sep_before != ";" or not segments:
        return "empty command"
    else:
        segments[-1].sep_after = None
    return segments


def _drop_trailing_fd(seg: _Segment) -> None:
    if seg.words and seg.words[-1].isdigit() and len(seg.words) > 1:
        seg.words.pop()


def _check_path(
    tok: str, cwd: Path, jail: WorkspaceJail, secrets: SecretMatcher, hits: list[str]
) -> str | None:
    """``tok`` が workspace 内へ解決されるか検査し、秘密該当を ``hits`` へ積む。"""
    if tok.startswith("~") or "=~" in tok or ":~" in tok:
        return f"tilde expansion: {tok}"
    if _BRACE_EXPANSION.search(tok):
        return f"brace expansion: {tok}"
    if any(c in tok for c in _GLOB_CHARS):
        targets = glob.glob(tok, root_dir=str(cwd)) or [tok]
    else:
        targets = [tok]
    for t in targets:
        candidate = Path(t) if Path(t).is_absolute() else cwd / t
        try:
            resolved = jail.resolve(candidate)
        except PathEscapeError:
            return f"outside workspace: {t}"
        rel = "" if resolved == jail.root else resolved.relative_to(jail.root).as_posix()
        if secrets.matches(rel) or secrets.matches(os.path.basename(t)):
            hits.append(rel or t)
    return None


def _option_values(args: list[str]) -> Iterator[str]:
    """``--`` より前のオプションに付いた値（``--opt=value`` と ``-ovalue``）。"""
    for tok in args:
        if tok == "--":
            return
        if tok.startswith("--"):
            if "=" in tok:
                yield tok.split("=", 1)[1]
        elif tok.startswith("-") and len(tok) > 2:
            yield tok[2:]


def _path_candidates(args: list[str]) -> Iterator[str]:
    """先頭語以降のパス候補（オプションに付いた値・``--`` 以降の全トークンを含む）。"""
    yield from _option_values(args)
    after_dashdash = False
    for tok in args:
        if after_dashdash:
            yield tok
            continue
        if tok == "--":
            after_dashdash = True
            continue
        if tok == "-" or tok.startswith("-"):
            continue
        yield tok


def _apply_cd(seg: _Segment, cwd: Path, jail: WorkspaceJail) -> tuple[Path | None, str | None]:
    """``cd`` を検査し、以降のセグメントが基準にする cwd を返す。"""
    if seg.sep_before not in (None, ";", "&&") or seg.sep_after not in (None, ";", "&&"):
        return None, "cd combined with | or ||"
    if "CDPATH" in os.environ:
        return None, "CDPATH is set"
    args = seg.words[1:]
    flags = [t for t in args if t.startswith("-") and t != "-"]
    targets = [t for t in args if not (t.startswith("-") and t != "-")]
    if any(f not in ("-L", "-P") for f in flags) or len(targets) != 1 or targets[0] == "-":
        return None, "cd requires exactly one directory"
    tok = targets[0]
    if tok.startswith("~") or _BRACE_EXPANSION.search(tok) or any(c in tok for c in _GLOB_CHARS):
        return None, f"cd target not literal: {tok}"
    candidate = Path(tok) if Path(tok).is_absolute() else cwd / tok
    try:
        resolved = jail.resolve(candidate)
    except PathEscapeError:
        return None, f"outside workspace: {tok}"
    if not resolved.is_dir():
        return None, f"cd target is not a directory: {tok}"
    return resolved, None


_CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def evaluate_auto_allow(
    command: str, *, jail: WorkspaceJail, secrets: SecretMatcher
) -> AutoAllowVerdict:
    """``command`` が証明可能に read-only か判定する。"""
    if not command.strip():
        return _reject("empty command")
    if _CONTROL.search(command):
        return _reject("control character")
    if "$" in command or "`" in command:
        return _reject("expansion ($ or backtick)")
    segments = split_command_segments(command)
    if isinstance(segments, str):
        return _reject(segments)

    cwd = jail.root
    hits: list[str] = []
    for seg in segments:
        head = seg.words[0]
        if "=" in head and _ENV_NAME.match(head.split("=", 1)[0]):
            return _reject(f"environment assignment prefix: {head}")
        if "/" in head:
            return _reject(f"path-specified command: {head}")
        if head not in _AUTO_ALLOW_COMMANDS:
            return _reject(f"command not in read-only vocabulary: {head}")
        args = seg.words[1:]
        rule = _RULES.get(head)
        if rule is not None:
            why = rule.violation(args, cwd)
            if why is not None:
                return _reject(f"{head}: {why}")
        candidates: list[str] = list(seg.extra_paths)
        if head == "cd":
            new_cwd, why = _apply_cd(seg, cwd, jail)
            if new_cwd is None:
                return _reject(f"cd: {why}")
            cwd = new_cwd
        elif head == "git":
            paths, why = _git_candidates(args)
            if why is not None:
                return _reject(why)
            candidates.extend(paths)
        elif head in _PATH_ARG_COMMANDS:
            candidates.extend(_path_candidates(args))
        for tok in candidates:
            why = _check_path(tok, cwd, jail, secrets, hits)
            if why is not None:
                return _reject(why)
    if hits:
        unique = tuple(dict.fromkeys(hits))
        return AutoAllowVerdict(kind="secret", reason="secret file", paths=unique)
    return AutoAllowVerdict(kind="allow")


def is_provably_read_only(
    command: str, *, jail: WorkspaceJail, secrets: SecretMatcher | None = None
) -> bool:
    """:func:`evaluate_auto_allow` が ``allow`` を返すか（述語形）。"""
    matcher = secrets if secrets is not None else SecretMatcher(_DEFAULT_SECRET_PATTERNS)
    return evaluate_auto_allow(command, jail=jail, secrets=matcher).kind == "allow"
