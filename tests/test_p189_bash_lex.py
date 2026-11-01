"""リダイレクト抽出の走査器のテスト。"""

from __future__ import annotations

import pytest

from satsumaestro.security.bash_lex import iter_redirects


def _redirects(command: str) -> list[tuple[str, str]]:
    return list(iter_redirects(command))


@pytest.mark.parametrize(
    "command",
    [
        "grep -c '</html>' dashboard/index.html && grep -c '</body>' dashboard/index.html",
        'grep -o "</item>" feed.xml',
        'for t in html body div; do grep -c "<$t" f; grep -c "</$t" f; done',
        "python3 -c \"import re; print(re.findall(r'</div>', open('i.html').read()))\"",
        "echo 'a < b'",
        "grep -n 'if x > 3' app.py",
        "grep '>' notes.txt",
        'sed "s|<br>|<br/>|g" index.html',
        "grep -e'</p> x' index.html",
        'echo "say \\"</x>\\""',
        "echo \\< b",
        "echo x # > /etc/passwd",
        "ls;# > /etc/passwd",
        'cat <<<"</x>"',
        "cmd 2>&1",
        "cmd >&2",
        "cmd >&-",
        "cmd 3<&0",
        "cat a.txt | grep x; ls && echo ok || echo ng",
    ],
)
def test_operators_outside_redirection_are_ignored(command: str) -> None:
    assert _redirects(command) == []


@pytest.mark.parametrize(
    "command",
    [
        "python3 - <<'EOF'\n# it's fine\nprint(r'</div>')\nEOF",
        'cat <<"EOF"\n</body>\nEOF',
        "cat <<EOF\n> /etc/passwd\nEOF",
        "cat <<-EOF\n\t</div>\n\tEOF",
        "cat <<A <<B\na>\nA\n</b>\nB",
        "cat <<EOF\n</never-terminated>",
    ],
)
def test_heredoc_bodies_are_skipped(command: str) -> None:
    assert _redirects(command) == []


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("echo x > /etc/passwd", [(">", "/etc/passwd")]),
        ("echo x >/etc/passwd", [(">", "/etc/passwd")]),
        ("echo x 2>/etc/passwd", [(">", "/etc/passwd")]),
        ("echo>x", [(">", "x")]),
        ("echo x >> logs/out.txt", [(">>", "logs/out.txt")]),
        ("echo x >| f", [(">|", "f")]),
        ("cat <> f", [("<>", "f")]),
        ("wc -l < feed.xml", [("<", "feed.xml")]),
        ("cmd &> f", [("&>", "f")]),
        ("cmd &>> f", [("&>>", "f")]),
        ("echo x >&/etc/passwd", [(">&", "/etc/passwd")]),
        ("echo x >& out.txt", [(">&", "out.txt")]),
        ("echo x > '/etc/x'", [(">", "/etc/x")]),
        ("echo x >'/etc/x'", [(">", "/etc/x")]),
        ('echo x >"/etc/x y"', [(">", "/etc/x y")]),
        ("echo x > a\\ b", [(">", "a b")]),
        ("echo x >/etc/x;ls", [(">", "/etc/x")]),
        ("cmd 2>&1 >/etc/x", [(">", "/etc/x")]),
        ("cat <<EOF > /etc/out\nx\nEOF", [(">", "/etc/out")]),
        ("cat <<EOF\nx\nEOF\necho y > /etc/z", [(">", "/etc/z")]),
        ("echo '<<X'\necho pwned > /etc/passwd", [(">", "/etc/passwd")]),
        ("echo 'a\n<<X\n' > /etc/passwd", [(">", "/etc/passwd")]),
        ("diff <(ls) <(cat < /etc/passwd)", [("<", "/etc/passwd")]),
        ("cat <(echo y > /etc/z)", [(">", "/etc/z")]),
        ("grep -c '</body>' index.html > count.txt", [(">", "count.txt")]),
    ],
)
def test_real_redirects_are_found(command: str, expected: list[tuple[str, str]]) -> None:
    assert _redirects(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "(( x = 1 << 2 ))\necho pwned > /etc/x\n2",
        "for (( i=0; i < 1<<1; i++ )); do :; done\necho pwned > /etc/x\n1",
        "echo ${x:-<<Z}\necho pwned > /etc/x\nZ",
        "echo $[1<<2]\necho pwned > /etc/x\n2]",
        "echo `echo <<Z`\necho pwned > /etc/x\nZ",
        "cat <<EOF\nEO\\\nF\necho pwned > /etc/x\nEOF",
        "echo a\\ #b > /etc/x",
        "echo $'\\'' > /etc/x #'",
    ],
)
def test_skipping_never_hides_a_real_redirect(command: str) -> None:
    assert (">", "/etc/x") in _redirects(command)


def test_quoted_heredoc_delimiter_does_not_join_continuations() -> None:
    assert _redirects("cat <<'EOF'\nEO\\\nF\n> /etc/x\nEOF") == []


def test_ansi_c_quoting() -> None:
    assert _redirects("cat <<<$'a\\nb'") == []
    with pytest.raises(ValueError, match="ANSI-C"):
        _redirects("echo x > $'\\x2fetc\\x2fpasswd'")


@pytest.mark.parametrize("command", ["echo don't", 'echo "abc', "echo x > 'unterminated"])
def test_unbalanced_quote_raises(command: str) -> None:
    with pytest.raises(ValueError, match="No closing quotation"):
        _redirects(command)
