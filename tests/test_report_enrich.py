"""Tests for report.json enrichment (metric_status + plausibility patches)."""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _sample_report():
    return {
        "portfolio": {
            "sharpe": 110.62,
            "sortino": None,
            "upi": None,
            "cagr": 176.24,
            "ulcer_index": 0.0,
            "calmar": 547762.0,
            "max_drawdown": 0.0003,
        },
        "plausibility_checks": [
            {"metric": "cagr", "ok": False, "severity": "high",
             "actual": 176.24, "reason": "cagr=176.24 outside [-0.99, 5]"},
            {"metric": "sharpe", "ok": False, "severity": "high",
             "actual": 110.62, "reason": "sharpe=110.621 outside [-5, 10]"},
            {"metric": "sortino", "ok": True, "severity": "ok",
             "actual": None, "reason": "undefined"},
            {"metric": "ulcer_index", "ok": True, "severity": "ok",
             "actual": 0.0, "reason": "within bounds"},
            {"metric": "upi", "ok": True, "severity": "ok",
             "actual": None, "reason": "undefined"},
        ],
        "plausibility": {"status": "implausible", "counts": {"high": 2, "low": 0, "ok": 3},
                         "failed": ["cagr=176.24 outside [-0.99, 5]",
                                    "sharpe=110.621 outside [-5, 10]"]},
    }


def test_metric_status_marks_undefined_vs_defined():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from report import _metric_status

    st = _metric_status(_sample_report()["portfolio"])
    assert st["sortino"]["state"] == "undefined"
    assert "downside" in st["sortino"]["reason"]
    assert st["upi"]["state"] == "undefined"
    assert st["sharpe"]["state"] == "defined"
    assert st["cagr"]["state"] == "defined"
    assert st["cagr"]["flag"] == "out_of_bounds"
    assert st["ulcer_index"]["state"] == "defined"
    assert "zero drawdown" in st["ulcer_index"]["note"]


def test_enrich_report_flags_zero_ulcer_and_rewrites_checks():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from report import enrich_report

    r = enrich_report(_sample_report())
    assert "metric_status" in r["portfolio"]
    ui_check = next(c for c in r["plausibility_checks"] if c["metric"] == "ulcer_index")
    assert ui_check["ok"] is False
    assert ui_check["severity"] == "low"
    assert "suspicious" in ui_check["reason"]
    assert r["plausibility"]["counts"]["low"] >= 1
    assert any("ulcer_index=0" in f for f in r["plausibility"]["failed"])


def test_enrich_report_json_serializes_null_status():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from report import enrich_report
    from dirty_fin_reports.simple.report import write_report

    r = enrich_report(_sample_report())
    path = REPO / "logs" / "test_enrich_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_report(r, path)
    loaded = json.loads(path.read_text())
    assert loaded["portfolio"]["sortino"] is None
    assert loaded["portfolio"]["metric_status"]["sortino"]["state"] == "undefined"
    path.unlink(missing_ok=True)
