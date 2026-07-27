from __future__ import annotations


def test_locked_analysis_parameters(config):
    assert config.master_seed == 20260715
    assert config.section("statistics")["bootstrap_replicates"] == 10_000
    assert config.section("statistics")["permutation_replicates"] == 10_000
    assert config.section("calibration")["bins"] == 5
    assert config.section("special_questions")["expected_questions"] == 15
    assert config.section("design")["primary_horizon"] == "T_24H"


def test_sqlite_and_public_export_are_isolated_from_rq_modules(project_root):
    analyses = project_root / "src" / "soccerarena_analysis" / "analyses"
    for path in analyses.glob("*.py"):
        text = path.read_text(encoding="utf-8").casefold()
        assert "sqlite3" not in text
        assert "public_csv" not in text


def test_single_website_csv_reader(project_root):
    source = project_root / "src" / "soccerarena_analysis"
    token = "resolve_path(" + chr(34) + "public_csv" + chr(34) + ")"
    readers = [
        path.name for path in source.rglob("*.py") if token in path.read_text(encoding="utf-8")
    ]
    assert readers == ["reconcile_public_export.py"]


def test_prohibited_rank_test_absent(project_root):
    source = project_root / "src" / "soccerarena_analysis"
    term = "wilco" + "xon"
    assert not [
        path for path in source.rglob("*.py") if term in path.read_text(encoding="utf-8").casefold()
    ]
