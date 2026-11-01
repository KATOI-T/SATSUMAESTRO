"""ネイティブの Windows では、起動の入口で WSL2 を案内して止める。"""

from __future__ import annotations

import pytest

from satsumaestro.cli import app as cli_app


def test_windows_stops_with_the_wsl2_notice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    started: list[bool] = []
    monkeypatch.setattr(cli_app.sys, "platform", "win32")
    monkeypatch.setattr(cli_app, "app", lambda: started.append(True))
    with pytest.raises(SystemExit) as exc:
        cli_app.main()
    assert exc.value.code == 1
    assert started == []
    err = capsys.readouterr().err
    assert "WSL2" in err and "wsl --install" in err


def test_other_platforms_start_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[bool] = []
    monkeypatch.setattr(cli_app.sys, "platform", "linux")
    monkeypatch.setattr(cli_app, "app", lambda: started.append(True))
    cli_app.main()
    assert started == [True]
