import io
import json
import urllib.error
from pathlib import Path

import pytest

from pariksha import cli, explain
from pariksha.findings import Finding
from pariksha.scan import scan
from pariksha.source import load

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_examples_leaky_and_clean():
    leaky = scan([EXAMPLES / "leaky_project"])
    assert {f.rule for f in leaky.findings} == {
        "PX001", "PX002", "PX003", "PX004", "PX005", "PX006", "PX007"}
    assert scan([EXAMPLES / "clean_project"]).findings == []


def test_notebook_locations_use_cells():
    src = load(EXAMPLES / "leaky_project" / "churn.ipynb")
    (finding,) = [f for f in scan([src.path]).findings if f.rule == "PX001"]
    assert finding.where.endswith("churn.ipynb [cell 3]:4")
    assert "(cell 4 line 1)" in finding.message


def test_notebook_cell_with_ipython_syntax_is_skipped(tmp_path):
    nb = {"cells": [
        {"cell_type": "code", "source": ["files = !ls\n", "x ="]},
        {"cell_type": "code", "source": ["model.fit(X_test, y_test)\n"]},
    ], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}
    path = write(tmp_path, "nb.ipynb", json.dumps(nb))
    (finding,) = scan([path]).findings
    assert finding.where.endswith("nb.ipynb [cell 2]:1")


def test_inline_ignore(tmp_path):
    path = write(tmp_path, "a.py",
                 "model.fit(X_test, y_test)  # pariksha: ignore[PX006]\n"
                 "model.fit(X_test, y_test)  # pariksha: ignore[PX001]\n"
                 "model.fit(X_test, y_test)  # pariksha: ignore\n")
    assert [f.line for f in scan([path]).findings] == [2]


def test_project_wide_seed_check(tmp_path):
    write(tmp_path, "train.py", "loss.backward()\n")
    assert [f.rule for f in scan([tmp_path]).findings] == ["PX004"]
    write(tmp_path, "seed.py", "torch.manual_seed(0)\n")
    assert scan([tmp_path]).findings == []


def test_unparseable_file_is_reported_not_fatal(tmp_path):
    write(tmp_path, "old.py", "print 'python 2'\n")
    result = scan([tmp_path])
    assert result.files == 0 and len(result.skipped) == 1


def test_exit_codes_and_json(tmp_path, capsys):
    path = write(tmp_path, "a.py", "a = train_test_split(X, y)\n")  # one low finding
    assert cli.main([str(path)]) == 0
    assert cli.main([str(path), "--fail-on", "low"]) == 1
    capsys.readouterr()
    assert cli.main([str(path), "--json", "--fail-on", "none"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["files_scanned"] == 1
    assert data["findings"][0]["rule"] == "PX004"
    assert cli.main([str(path), "--select", "PX999"]) == 2
    assert cli.main([str(tmp_path / "missing.py")]) == 2


def test_explain_without_key_still_reports(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("BHARATCODE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    path = write(tmp_path, "a.py", "model.fit(X_test, y_test)\n")
    assert cli.main([str(path), "--explain", "--no-colour"]) == 1
    captured = capsys.readouterr()
    assert "BHARATCODE_API_KEY" in captured.err
    assert "PX006" in captured.out


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_explain_sends_finding_and_strips_thinking(tmp_path, monkeypatch):
    path = write(tmp_path, "a.py", "import x\nmodel.fit(X_test, y_test)\n")
    finding = Finding("PX006", "high", 2, "trained on test", load(path))
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["auth"] = request.headers["Authorization"]
        sent["body"] = json.loads(request.data)
        reply = {"choices": [{"message": {"content": "<think>hmm</think>\nUse a validation set."}}]}
        return FakeResponse(json.dumps(reply).encode())

    monkeypatch.setattr(explain.urllib.request, "urlopen", fake_urlopen)
    text = explain.explain(finding, api_key="k", base_url="https://example.test/v1", model="m")
    assert text == "Use a validation set."
    assert sent["url"] == "https://example.test/v1/chat/completions"
    assert sent["auth"] == "Bearer k"
    assert sent["body"]["model"] == "m"
    assert "> " in sent["body"]["messages"][1]["content"]
    assert "model.fit(X_test, y_test)" in sent["body"]["messages"][1]["content"]


def test_explain_http_error(tmp_path, monkeypatch):
    path = write(tmp_path, "a.py", "model.fit(X_test, y_test)\n")
    finding = Finding("PX006", "high", 1, "trained on test", load(path))

    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {},
                                     io.BytesIO(b'{"error":"invalid key"}'))

    monkeypatch.setattr(explain.urllib.request, "urlopen", fail)
    with pytest.raises(explain.ExplainError, match="HTTP 401"):
        explain.explain(finding, api_key="bad")


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    env = write(tmp_path, ".env", "# comment\nBHARATCODE_MODEL='from-file'\nBHARATCODE_API_KEY=abc\n")
    environ = {"BHARATCODE_MODEL": "from-env"}
    monkeypatch.setattr(explain.os, "environ", environ)
    explain.load_dotenv(env)
    assert environ == {"BHARATCODE_MODEL": "from-env", "BHARATCODE_API_KEY": "abc"}
