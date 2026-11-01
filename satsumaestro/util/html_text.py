"""HTML → 本文抽出 → Markdown の軽量実装。"""

from __future__ import annotations

import html as html_mod
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "nav",
        "footer",
        "aside",
        "iframe",
        "svg",
        "form",
        "button",
        "select",
        "head",
    }
)
_VOID_TAGS = frozenset(
    {"br", "hr", "img", "meta", "link", "input", "area", "base", "col", "embed", "source", "wbr"}
)
_CONTAINER_TAGS = frozenset({"article", "main", "section", "div", "html", "body", "td"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "body",
        "ul",
        "ol",
        "li",
        "table",
        "tr",
        "blockquote",
        "pre",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
    }
)

_MAIN_TEXT_RATIO = 0.6

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    text: str = ""
    children: list[_Node] = field(default_factory=list)

    def text_len(self) -> int:
        if self.tag == "":
            return len(self.text.strip())
        return sum(c.text_len() for c in self.children)


class _TreeBuilder(HTMLParser):
    """壊れた HTML にも耐える寛容なツリー構築（未対応の閉じタグは無視）。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node(tag="root")
        self._stack: list[_Node] = [self.root]
        self._drop_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if self._drop_depth > 0:
            if tag in _DROP_TAGS and tag not in _VOID_TAGS:
                self._drop_depth += 1
            return
        if tag in _DROP_TAGS:
            if tag not in _VOID_TAGS:
                self._drop_depth += 1
            return
        node = _Node(tag=tag, attrs={k: (v or "") for k, v in attrs})
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if self._drop_depth > 0:
            if tag in _DROP_TAGS:
                self._drop_depth -= 1
            return
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._drop_depth > 0 or not data:
            return
        self._stack[-1].children.append(_Node(tag="", text=data))


def _find_main(root: _Node) -> _Node:
    """本文コンテナを選ぶ（簡易 readability）。"""
    for tag in ("article", "main"):
        found = _find_first(root, tag)
        if found is not None and found.text_len() > 0:
            return found
    total = root.text_len()
    if total == 0:
        return root
    best = root
    node = root
    while True:
        candidates = [
            c
            for c in node.children
            if c.tag in _CONTAINER_TAGS and c.text_len() >= total * _MAIN_TEXT_RATIO
        ]
        if not candidates:
            break
        node = max(candidates, key=lambda c: c.text_len())
        best = node
    return best


def _find_first(node: _Node, tag: str) -> _Node | None:
    if node.tag == tag:
        return node
    for c in node.children:
        found = _find_first(c, tag)
        if found is not None:
            return found
    return None


class _MarkdownWriter:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self._open_lists: list[str] = []
        self._ol_counters: list[int] = []

    def render(self, node: _Node) -> None:
        for c in node.children:
            self._emit(c)

    def _emit(self, node: _Node) -> None:
        if node.tag == "":
            self.parts.append(_WS_RE.sub(" ", node.text.replace("\n", " ")))
            return
        tag = node.tag
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append(f"\n\n{'#' * int(tag[1])} {self._inline(node).strip()}\n\n")
            return
        if tag == "pre":
            code = _collect_text(node)
            self.parts.append(f"\n\n```\n{code.rstrip()}\n```\n\n")
            return
        if tag == "code":
            self.parts.append(f"`{_collect_text(node).strip()}`")
            return
        if tag == "a":
            text = self._inline(node).strip()
            href = node.attrs.get("href", "")
            if href and not href.startswith(("#", "javascript:")) and text:
                self.parts.append(f"[{text}]({href})")
            else:
                self.parts.append(text)
            return
        if tag in {"strong", "b"}:
            self.parts.append(f"**{self._inline(node).strip()}**")
            return
        if tag in {"em", "i"}:
            self.parts.append(f"*{self._inline(node).strip()}*")
            return
        if tag == "br":
            self.parts.append("\n")
            return
        if tag in {"ul", "ol"}:
            self._open_lists.append(tag)
            self._ol_counters.append(0)
            self.parts.append("\n")
            self.render(node)
            self._open_lists.pop()
            self._ol_counters.pop()
            self.parts.append("\n")
            return
        if tag == "li":
            indent = "  " * (len(self._open_lists) - 1) if self._open_lists else ""
            if self._open_lists and self._open_lists[-1] == "ol":
                self._ol_counters[-1] += 1
                marker = f"{self._ol_counters[-1]}."
            else:
                marker = "-"
            self.parts.append(f"\n{indent}{marker} ")
            self.render(node)
            return
        if tag == "blockquote":
            inner = self._inline(node).strip()
            quoted = "\n".join(f"> {line}" for line in inner.splitlines())
            self.parts.append(f"\n\n{quoted}\n\n")
            return
        if tag == "tr":
            cells = [self._inline(c).strip() for c in node.children if c.tag in {"td", "th"}]
            if cells:
                self.parts.append("\n| " + " | ".join(cells) + " |")
            return
        if tag == "img":
            alt = node.attrs.get("alt", "").strip()
            if alt:
                self.parts.append(f"({alt})")
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
            self.render(node)
            self.parts.append("\n\n")
            return
        self.render(node)

    def _inline(self, node: _Node) -> str:
        sub = _MarkdownWriter()
        sub._open_lists = self._open_lists
        sub._ol_counters = self._ol_counters
        sub.render(node)
        return "".join(sub.parts)


def _collect_text(node: _Node) -> str:
    if node.tag == "":
        return node.text
    return "".join(_collect_text(c) for c in node.children)


def html_to_markdown(html: str) -> str:
    """HTML から本文を抽出し Markdown を返す。"""
    builder = _TreeBuilder()
    try:
        builder.feed(html)
        builder.close()
    except Exception:
        return ""
    main = _find_main(builder.root)
    writer = _MarkdownWriter()
    writer.render(main)
    body = "".join(writer.parts)
    lines = [line.strip() for line in body.splitlines()]
    text = _BLANK_RE.sub("\n\n", "\n".join(lines)).strip()
    if not text:
        return ""
    title = html_mod.unescape(_WS_RE.sub(" ", builder.title)).strip()
    if title:
        return f"# {title}\n\n{text}"
    return text
