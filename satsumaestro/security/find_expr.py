"""find の式が、あるファイルにそれを変える動作（-delete・-exec のコマンド）を届かせ得るかを読む。"""

from __future__ import annotations

import fnmatch
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

_NO_ARGS = frozenset(
    {
        "-d",
        "-daystart",
        "-delete",
        "-depth",
        "-empty",
        "-executable",
        "-false",
        "-follow",
        "-ignore_readdir_race",
        "-ls",
        "-mount",
        "-noignore_readdir_race",
        "-noleaf",
        "-nogroup",
        "-nouser",
        "-nowarn",
        "-print",
        "-print0",
        "-prune",
        "-quit",
        "-readable",
        "-true",
        "-warn",
        "-writable",
        "-xdev",
    }
)
_TWO_ARGS = frozenset({"-fprintf"})
_OPTIONS = frozenset({"-maxdepth", "-mindepth", "-regextype"})
_EXEC = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
_NAME_TESTS = {"-name": False, "-iname": True}
_PATH_TESTS = {"-path": False, "-ipath": True, "-wholename": False, "-iwholename": True}
_TYPES: dict[str, Callable[[int], bool]] = {
    "f": stat.S_ISREG,
    "d": stat.S_ISDIR,
    "l": stat.S_ISLNK,
    "p": stat.S_ISFIFO,
    "s": stat.S_ISSOCK,
    "c": stat.S_ISCHR,
    "b": stat.S_ISBLK,
}
_MAX_ENTRIES = 200_000
_GLOB = frozenset("*?[\\")

_Node = tuple[object, ...]


class _Unreadable(Exception):
    """読めない式。"""


@dataclass(frozen=True)
class FindEntry:
    """find が見るファイル 1 つ（``shown`` は find が出す形、``kind`` は一覧で分かった種類）。"""

    path: str
    shown: str
    depth: int
    kind: str = ""

    def child(self, name: str, kind: str = "") -> FindEntry:
        shown = self.shown + name if self.shown.endswith("/") else f"{self.shown}/{name}"
        return FindEntry(os.path.join(self.path, name), shown, self.depth + 1, kind)


@dataclass
class _Visit:
    below: bool = False
    changes: bool = False
    pruned: bool = False


class FindExpression:
    """find の式（起点の後ろの語）。読めない所は、当たり得るとみなす。"""

    def __init__(
        self,
        tokens: Sequence[str],
        exec_changes: Callable[[list[str]], bool],
        *,
        follow: bool = False,
        depth: bool = False,
        deleting: bool = True,
    ) -> None:
        # deleting なら -delete も変更に数え、名前が . や .. の起点は変わらないとみなす
        self._tokens = list(tokens)
        self._exec_changes = exec_changes
        self._deleting = deleting
        self.follow = follow or "-follow" in self._tokens
        self._max = _depth_option(self._tokens, "-maxdepth")
        self._min = _depth_option(self._tokens, "-mindepth") or 0
        # -delete と -depth は深い方から見るので -prune が効かない
        self._prunes = not depth and not any(t in ("-delete", "-depth", "-d") for t in tokens)
        self._tree: _Node | None
        try:
            node, end = self._list(0)
            self._tree = node if end == len(self._tokens) else None
        except _Unreadable:
            self._tree = None

    @property
    def acts(self) -> bool:
        """ファイルを変える動作を含むか（読めない式は含むとみなす）。"""
        return self._tree is None or self._has_action(self._tree)

    def first_changed(self, written: str, root: Path, start: Path) -> Path | None:
        """起点 ``root``（書いた形 ``written``）の下の ``start`` とその中で、変更が届き得る物。"""
        if self._tree is None:
            return start
        entry = FindEntry(str(root), written, 0)
        for k, part in enumerate(start.relative_to(root).parts):
            reach = self._reach(entry)
            if reach is False:
                return None
            if reach and not (k == 0 and self._deleting and _dot_named(written)):
                return Path(entry.path)
            entry = entry.child(part)
        try:
            is_dir = stat.S_ISDIR((os.stat if self.follow else os.lstat)(entry.path).st_mode)
        except OSError:
            is_dir = False
        pending = [(entry, is_dir)]
        seen: set[tuple[int, int]] = set()
        for _ in range(_MAX_ENTRIES):
            if not pending:
                return None
            entry, is_dir = pending.pop()
            reach = self._reach(entry)
            if reach is True:
                return Path(entry.path)
            if reach is None and is_dir and self._visit(entry, below=True).changes:
                pending.extend(self._children(entry, seen))
        return start

    def _reach(self, entry: FindEntry) -> bool | None:
        """変更が届くなら真、find がその中へ入らないなら偽、どちらでもなければ ``None``。"""
        if self._max is not None and entry.depth >= self._max:
            return entry.depth == self._max >= self._min and self._visit(entry).changes
        if entry.depth < self._min:
            return None
        visit = self._visit(entry)
        if visit.changes:
            return True
        return False if self._prunes and visit.pruned else None

    def _children(
        self, entry: FindEntry, seen: set[tuple[int, int]]
    ) -> list[tuple[FindEntry, bool]]:
        try:
            if self.follow:
                info = os.stat(entry.path)
                if (info.st_dev, info.st_ino) in seen:
                    return []
                seen.add((info.st_dev, info.st_ino))
            with os.scandir(entry.path) as listing:
                found = [(item.name, *self._kind(item)) for item in listing]
        except OSError:
            return []
        return [(entry.child(name, kind), is_dir) for name, kind, is_dir in sorted(found)]

    def _kind(self, item: os.DirEntry[str]) -> tuple[str, bool]:
        """一覧の項目の種類（f・d・l、分からなければ空）と、中へ入るか。"""
        try:
            if item.is_symlink():
                return "l", self.follow and item.is_dir()
            if item.is_dir(follow_symlinks=False):
                return "d", True
            return ("f" if item.is_file(follow_symlinks=False) else ""), False
        except OSError:
            return "", False

    def _has_action(self, node: _Node) -> bool:
        if node[0] == "test":
            return self._changes(str(node[1]), _args(node))
        return any(self._has_action(_child(node, k)) for k in range(1, len(node)))

    def _changes(self, name: str, args: list[str]) -> bool:
        if name == "-delete":
            return self._deleting
        return name in _EXEC and bool(args) and self._exec_changes(args)

    def _visit(self, entry: FindEntry, *, below: bool = False) -> _Visit:
        """``below`` なら、``entry`` の中のどれかに当たり得るか（パスの条件だけで決める）。"""
        visit = _Visit(below=below)
        assert self._tree is not None
        self._eval(self._tree, entry, True, visit)
        return visit

    def _eval(self, node: _Node, entry: FindEntry, sure: bool, visit: _Visit) -> bool | None:
        kind = node[0]
        if kind == "not":
            value = self._eval(_child(node, 1), entry, sure, visit)
            return None if value is None else not value
        if kind == "test":
            return self._test(str(node[1]), _args(node), entry, sure, visit)
        left = self._eval(_child(node, 1), entry, sure, visit)
        if kind == "list":
            return self._eval(_child(node, 2), entry, sure, visit)
        if left is (kind == "or"):
            return left
        right = self._eval(_child(node, 2), entry, sure and left is not None, visit)
        if right is (kind == "or"):
            return right
        return None if None in (left, right) else kind == "and"

    def _test(
        self, name: str, args: list[str], entry: FindEntry, sure: bool, visit: _Visit
    ) -> bool | None:
        if name in _PATH_TESTS:
            fold = _PATH_TESTS[name]
            if visit.below:
                return _matches_below(entry.shown, args[0], fold=fold)
            return _matches(entry.shown, args[0], fold=fold)
        if visit.below and (name in _NAME_TESTS or name in ("-type", "-empty")):
            return None
        if name in _NAME_TESTS:
            base = os.path.basename(entry.shown.rstrip("/")) or entry.shown
            return _matches(base, args[0], fold=_NAME_TESTS[name])
        if name == "-type":
            return _type_is(entry, args[0], follow=self.follow)
        if name == "-empty":
            return _empty(entry.path, follow=self.follow)
        if name == "-false":
            return False
        if name == "-prune":
            visit.pruned = visit.pruned or (sure and not visit.below)
            return True
        if self._changes(name, args):
            visit.changes = True
            return True
        if name in _EXEC:
            return None
        if name in _NO_ARGS or name in _OPTIONS:
            return True
        return None

    def _list(self, i: int) -> tuple[_Node, int]:
        node, i = self._or(i)
        while i < len(self._tokens) and self._tokens[i] == ",":
            right, i = self._or(i + 1)
            node = ("list", node, right)
        return node, i

    def _or(self, i: int) -> tuple[_Node, int]:
        node, i = self._and(i)
        while i < len(self._tokens) and self._tokens[i] in ("-o", "-or"):
            right, i = self._and(i + 1)
            node = ("or", node, right)
        return node, i

    def _and(self, i: int) -> tuple[_Node, int]:
        node, i = self._unary(i)
        while i < len(self._tokens) and self._tokens[i] not in ("-o", "-or", ",", ")"):
            if self._tokens[i] in ("-a", "-and"):
                i += 1
            right, i = self._unary(i)
            node = ("and", node, right)
        return node, i

    def _unary(self, i: int) -> tuple[_Node, int]:
        if i >= len(self._tokens):
            raise _Unreadable
        token = self._tokens[i]
        if token in ("!", "-not"):
            inner, i = self._unary(i + 1)
            return ("not", inner), i
        if token == "(":
            inner, i = self._list(i + 1)
            if i >= len(self._tokens) or self._tokens[i] != ")":
                raise _Unreadable
            return inner, i + 1
        if token in _EXEC:
            # 区切りの ; は単純コマンドに分けたときに落ちるので、無ければ最後までを引数にする
            end = next(
                (j for j in range(i + 1, len(self._tokens)) if self._tokens[j] in (";", "+")),
                len(self._tokens),
            )
            return ("test", token, self._tokens[i + 1 : end]), min(end + 1, len(self._tokens))
        if not token.startswith("-"):
            raise _Unreadable
        width = 0 if token in _NO_ARGS else 2 if token in _TWO_ARGS else 1
        args = self._tokens[i + 1 : i + 1 + width]
        if len(args) < width:
            raise _Unreadable
        return ("test", token, args), i + 1 + width


def _child(node: _Node, at: int) -> _Node:
    child = node[at]
    assert isinstance(child, tuple)
    return child


def _args(node: _Node) -> list[str]:
    args = node[2]
    assert isinstance(args, list)
    return args


def _dot_named(written: str) -> bool:
    return os.path.basename(written.rstrip("/")) in (".", "..")


def _depth_option(tokens: list[str], option: str) -> int | None:
    value = None
    for i, token in enumerate(tokens[:-1]):
        if token == option and tokens[i + 1].isdigit():
            value = int(tokens[i + 1])
    return value


def _matches(value: str, pattern: str, *, fold: bool) -> bool:
    if fold:
        return fnmatch.fnmatchcase(value.casefold(), pattern.casefold())
    return fnmatch.fnmatchcase(value, pattern)


def _matches_below(shown: str, pattern: str, *, fold: bool) -> bool | None:
    """``shown`` の中のパスすべてについて、``pattern`` に当たるか（決まらなければ ``None``）。"""
    prefix = shown if shown.endswith("/") else shown + "/"
    if fold:
        prefix, pattern = prefix.casefold(), pattern.casefold()
    head = next((pattern[:i] for i, ch in enumerate(pattern) if ch in _GLOB), pattern)
    if not (prefix.startswith(head) or head.startswith(prefix)):
        return False
    # 中のパスは prefix の後に何かが続くので、head の後が * だけなら必ず当たる
    return True if pattern == head + "*" and prefix.startswith(head) else None


def _type_is(entry: FindEntry, letters: str, *, follow: bool) -> bool | None:
    checks = [_TYPES.get(letter) for letter in letters.split(",")]
    if any(check is None for check in checks):
        return None
    if entry.kind and not (follow and entry.kind == "l"):
        return entry.kind in letters.split(",")
    try:
        mode = (os.stat(entry.path) if follow else os.lstat(entry.path)).st_mode
    except OSError:
        try:
            mode = os.lstat(entry.path).st_mode
        except OSError:
            return None
    return any(check(mode) for check in checks if check is not None)


def _empty(path: str, *, follow: bool) -> bool | None:
    try:
        info = os.stat(path) if follow else os.lstat(path)
        if stat.S_ISDIR(info.st_mode):
            return not os.listdir(path)
        return stat.S_ISREG(info.st_mode) and info.st_size == 0
    except OSError:
        return None
