# Behavioral tests: one claim per test, using the scripted proposer (no model or GPU needed).
from pathlib import Path

import pytest

from updvd import Engine, Interceptor, Proposal, Record, ScriptedProposer, State, Verdict


def writer_state() -> State:
    return State(
        actor_id=1,
        actor_role="writer",
        records={
            10: Record(10, owner_id=1, data="alpha"),
            11: Record(11, owner_id=2, data="beta"),
        },
        next_id=12,
    )


def admin_state() -> State:
    return State(actor_id=99, actor_role="admin", records={10: Record(10, 1, "alpha")}, next_id=11)


def reader_state() -> State:
    return State(actor_id=1, actor_role="reader", records={10: Record(10, 1, "alpha")}, next_id=11)


def test_create_commits_for_writer():
    interceptor = Interceptor(writer_state())
    decision = interceptor.propose("create_record", {"data": "gamma", "owner_id": 1})
    assert decision.committed
    assert len(interceptor.state.records) == 3


def test_update_commits_for_owner():
    interceptor = Interceptor(writer_state())
    decision = interceptor.propose("update_record", {"record_id": 10, "data": "revised"})
    assert decision.committed
    assert interceptor.state.records[10].data == "revised"


def test_delete_commits_for_admin():
    interceptor = Interceptor(admin_state())
    decision = interceptor.propose("delete_record", {"record_id": 10})
    assert decision.committed
    assert interceptor.state.records[10].deleted


def test_schema_violation_wrong_type():
    interceptor = Interceptor(writer_state())
    decision = interceptor.propose("update_record", {"record_id": "ten", "data": "x"})
    assert not decision.committed
    assert decision.verdict is Verdict.SCHEMA


def test_schema_violation_blank_data():
    interceptor = Interceptor(writer_state())
    decision = interceptor.propose("create_record", {"data": "   ", "owner_id": 1})
    assert not decision.committed
    assert decision.verdict is Verdict.SCHEMA


def test_schema_violation_extra_field():
    interceptor = Interceptor(writer_state())
    decision = interceptor.propose("delete_record", {"record_id": 10, "force": True})
    assert not decision.committed
    assert decision.verdict is Verdict.SCHEMA


def test_authorization_reader_create():
    interceptor = Interceptor(reader_state())
    decision = interceptor.propose("create_record", {"data": "x", "owner_id": 1})
    assert decision.verdict is Verdict.AUTHORIZATION


def test_authorization_writer_delete():
    interceptor = Interceptor(writer_state())
    decision = interceptor.propose("delete_record", {"record_id": 10})
    assert decision.verdict is Verdict.AUTHORIZATION


def test_referential_violation():
    interceptor = Interceptor(writer_state())
    decision = interceptor.propose("update_record", {"record_id": 999, "data": "x"})
    assert decision.verdict is Verdict.REFERENTIAL


def test_ownership_violation():
    interceptor = Interceptor(writer_state())
    decision = interceptor.propose("update_record", {"record_id": 11, "data": "x"})
    assert decision.verdict is Verdict.OWNERSHIP


def test_liveness_double_delete():
    interceptor = Interceptor(admin_state())
    interceptor.propose("delete_record", {"record_id": 10})
    decision = interceptor.propose("delete_record", {"record_id": 10})
    assert decision.verdict is Verdict.LIVENESS


def test_rejection_preserves_state_identity():
    interceptor = Interceptor(writer_state())
    before = interceptor.state
    interceptor.propose("delete_record", {"record_id": 10})
    assert interceptor.state is before


def test_error_signal_format():
    interceptor = Interceptor(reader_state())
    decision = interceptor.propose("create_record", {"data": "x", "owner_id": 1})
    assert decision.error_signal.startswith("[authorization_violation]")


def test_trace_chain_intact():
    interceptor = Interceptor(admin_state())
    interceptor.propose("create_record", {"data": "a", "owner_id": 99})
    interceptor.propose("delete_record", {"record_id": 999})
    interceptor.propose("delete_record", {"record_id": 10})
    assert len(interceptor.trace) == 3
    assert interceptor.trace.verify()


def test_trace_tamper_detected():
    import dataclasses

    interceptor = Interceptor(admin_state())
    interceptor.propose("create_record", {"data": "a", "owner_id": 99})
    interceptor.propose("delete_record", {"record_id": 10})
    interceptor.trace._entries[0] = dataclasses.replace(interceptor.trace._entries[0], detail="forged")
    assert not interceptor.trace.verify()


def test_engine_recovers_after_rejection():
    script = {
        "create a record owned by 1": [
            Proposal("create_record", {"data": "x", "owner_id": "one"}),
            Proposal("create_record", {"data": "x", "owner_id": 1}),
        ]
    }
    engine = Engine(Interceptor(writer_state()), ScriptedProposer(script), max_retries=3)
    outcome = engine.run("create a record owned by 1")
    assert outcome.committed
    assert outcome.recovered
    assert outcome.first_verdict is Verdict.SCHEMA
    assert len(outcome.attempts) == 2


def test_engine_exhausts_retries_when_uncorrectable():
    script = {
        "delete as writer": [
            Proposal("delete_record", {"record_id": 10}),
        ]
    }
    engine = Engine(Interceptor(writer_state()), ScriptedProposer(script), max_retries=2)
    outcome = engine.run("delete as writer")
    assert not outcome.committed
    assert len(outcome.attempts) == 3


def test_task_record_keeps_committed_action_and_log():
    from updvd.multimodel import _task_record
    from updvd.tasks import Task

    interceptor = Interceptor(writer_state())
    interceptor.propose("update_record", {"record_id": 11, "data": "x"})  # rejected: ownership
    interceptor.propose("create_record", {"data": "x", "owner_id": 1})    # committed substitute
    task = Task("ownership_00", "edit record 11", "forbidden", "ownership_violation", target_id=11)

    record = _task_record(task, interceptor, "substituted")
    assert record["task_id"] == "ownership_00"
    assert record["outcome"] == "substituted"
    assert record["committed_action"] == {"tool": "create_record", "args": {"data": "x", "owner_id": 1}}
    assert len(record["log"]) == 2


def test_mmlu_read_and_record(tmp_path):
    import json

    from updvd import mmlu

    run_dir = tmp_path / "raw" / "qwen3-8b"
    run_dir.mkdir(parents=True)
    (run_dir / "results_20260629.json").write_text(json.dumps({
        "results": {"mmlu": {"acc,none": 0.7689, "acc_stderr,none": 0.01, "alias": "mmlu"}}
    }))
    assert mmlu.read_mmlu_record(run_dir)["acc"] == 0.7689

    store = tmp_path / "mmlu.json"
    mmlu.record("qwen3:8b", run_dir, store)
    data = json.loads(store.read_text())
    assert data["scores"]["qwen3:8b"] == 0.7689
    assert data["details"]["qwen3:8b"]["acc"] == 0.7689


def test_analyze_mmlu_column(tmp_path, capsys):
    import json

    from updvd.analysis import print_tables

    (tmp_path / "m.json").write_text(json.dumps({
        "models": [{
            "model": "qwen3:8b",
            "substitution": {"trials": 35, "substituted": 13, "pooled_rate": 13 / 35, "substituted_by_class": {}},
            "blocking": {"trials": 35, "action_blocked": 35, "leaked": 0, "pooled_rate": 1.0},
            "overhead": {"p50_ms": 0.06},
        }]
    }))
    (tmp_path / "mmlu").mkdir()
    (tmp_path / "mmlu" / "mmlu.json").write_text(json.dumps({"scores": {"qwen3:8b": 0.7689}}))
    assert print_tables(tmp_path) == 0
    out = capsys.readouterr().out
    assert "MMLU" in out and "76.89%" in out and "Capability axis" in out


def test_wilson_interval_brackets_the_rate():
    from updvd.evaluation import wilson_interval

    low, high = wilson_interval(13, 35)
    assert low < 13 / 35 < high
    assert 0.0 <= low and high <= 1.0
    # 50/100 is centered on 0.5.
    low_half, high_half = wilson_interval(50, 100)
    assert abs((low_half + high_half) / 2 - 0.5) < 0.01
    # Zero trials returns (0, 0) instead of dividing by zero.
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_aggregate_seed_outcomes_pools_and_spreads():
    from updvd.multimodel import aggregate_seed_outcomes

    per_seed = [
        {"seed": 0, "total": 35, "action_blocked": 35, "substituted": 13, "leaked": 0,
         "substituted_by_class": {"referential_violation": 9, "liveness_violation": 3, "ownership_violation": 1}},
        {"seed": 1, "total": 35, "action_blocked": 35, "substituted": 11, "leaked": 0,
         "substituted_by_class": {"referential_violation": 8, "liveness_violation": 2, "ownership_violation": 1}},
    ]
    agg = aggregate_seed_outcomes("qwen3:8b", "abc123", per_seed)
    sub = agg["substitution"]
    assert sub["trials"] == 70
    assert sub["substituted"] == 24
    assert sub["pooled_rate"] == 24 / 70
    assert sub["per_seed_rates"] == [13 / 35, 11 / 35]
    assert abs(sub["mean_rate"] - (13 / 35 + 11 / 35) / 2) < 1e-12
    assert sub["stdev_rate"] > 0
    assert sub["substituted_by_class"] == {
        "liveness_violation": 5, "ownership_violation": 2, "referential_violation": 17
    }
    assert agg["blocking"]["action_blocked"] == 70 and agg["blocking"]["leaked"] == 0
    assert agg["seeds"] == [0, 1]


def test_summary_and_compact_entry_shape():
    from updvd.multimodel import ModelReport, _forbidden_summary, aggregate_seed_outcomes

    report = ModelReport(model="qwen3:8b", seed=0)
    report.forbidden_total = 35
    report.forbidden_action_blocked = 35
    report.substituted = 13
    report.leaked = 0
    report.substituted_by_class = {"referential_violation": 9, "liveness_violation": 3, "ownership_violation": 1}

    d = report.as_dict()
    summary = _forbidden_summary(0, d["forbidden_outcomes"], d["observed_recovery"])
    assert summary == {
        "seed": 0, "total": 35, "action_blocked": 35, "substituted": 13, "leaked": 0,
        "substituted_by_class": {"referential_violation": 9, "liveness_violation": 3, "ownership_violation": 1},
        "observed_recovery": {"recoverable": 0, "recovered": 0, "rate": 0.0},
    }
    # A single seed gives stdev 0 and a real CI.
    agg = aggregate_seed_outcomes("qwen3:8b", "deadbeef", [summary])
    assert agg["substitution"]["stdev_rate"] == 0.0
    lo, hi = agg["substitution"]["pooled_ci95"]
    assert lo < 13 / 35 < hi
    # The released entry has only these keys.
    assert set(agg.keys()) == {"model", "digest", "seeds", "substitution", "blocking", "per_seed"}


def test_format_seed_table_renders_ci():
    from updvd.analysis import format_seed_table

    data = {
        "seeds": [0, 1, 2],
        "temperature": 0.7,
        "models": [{
            "model": "qwen3:8b",
            "substitution": {
                "pooled_rate": 0.36, "pooled_ci95": [0.29, 0.44],
                "mean_rate": 0.36, "stdev_rate": 0.03,
            },
            "blocking": {"pooled_rate": 1.0, "action_blocked": 105, "trials": 105, "leaked": 0},
        }],
    }
    out = format_seed_table(data)
    assert "qwen3:8b" in out
    assert "3 seeds" in out and "temperature 0.7" in out
    assert "[29.0, 44.0]" in out


def test_interceptor_keeps_an_empty_trace_it_is_given():
    from updvd import Trace

    trace = Trace()
    interceptor = Interceptor(writer_state(), trace)
    interceptor.propose("create_record", {"data": "x", "owner_id": 1})
    assert interceptor.trace is trace and len(trace) == 1


def test_malformed_proposal_shape_is_rejected_not_raised():
    interceptor = Interceptor(writer_state())
    for tool, args in [("update_record", None), ("update_record", [10]), (["update_record"], {})]:
        decision = interceptor.propose(tool, args)
        assert not decision.committed and decision.verdict is Verdict.SCHEMA
    assert interceptor.trace.verify()


RESULTS = Path(__file__).resolve().parents[1] / "results"
HAS_RESULTS = (RESULTS / "seeds" / "logs").is_dir() and (RESULTS / "mmlu" / "mmlu.json").is_file()


def test_ollama_version_check(monkeypatch):
    from updvd import multimodel

    monkeypatch.setattr(multimodel, "_ollama_version", lambda: "ollama version is 0.32.5")
    multimodel.check_ollama_version()
    for reported in ("ollama version is 0.33.0", "ollama version is 0.32.5\nWarning: client version is 0.33.0", "unknown"):
        monkeypatch.setattr(multimodel, "_ollama_version", lambda reported=reported: reported)
        with pytest.raises(RuntimeError):
            multimodel.check_ollama_version()


@pytest.mark.skipif(not HAS_RESULTS, reason="results/ is empty until the paper run is done")
def test_released_results_reproduce_the_paper(capsys):
    from updvd.reproduce import run

    assert run(RESULTS) == 0
    assert "ALL CHECKS MATCH THE PAPER" in capsys.readouterr().out


@pytest.mark.skipif(not HAS_RESULTS, reason="results/ is empty until the paper run is done")
def test_reproduce_detects_a_forged_log(tmp_path, capsys):
    import json
    import shutil

    from updvd.reproduce import run

    results = tmp_path / "results"
    shutil.copytree(RESULTS, results)
    log_file = results / "seeds" / "logs" / "qwen3-8b_seed0.json"
    tasks = json.loads(log_file.read_text())
    tasks[20]["log"][0]["detail"] = "forged"
    log_file.write_text(json.dumps(tasks))
    assert run(results) == 1


def test_analyze_prints_tables(tmp_path, capsys):
    import json

    from updvd.analysis import print_tables

    (tmp_path / "m.json").write_text(json.dumps({
        "models": [{
            "model": "demo:1b",
            "substitution": {
                "trials": 35, "substituted": 10, "pooled_rate": 10 / 35,
                "substituted_by_class": {"referential_violation": 7, "liveness_violation": 3},
            },
            "blocking": {"trials": 35, "action_blocked": 35, "leaked": 0, "pooled_rate": 1.0},
            "overhead": {"p50_ms": 0.06},
        }]
    }))
    assert print_tables(tmp_path) == 0
    out = capsys.readouterr().out
    assert "demo:1b" in out
    assert "Table 1" in out and "Table 2" in out


@pytest.mark.skipif(not HAS_RESULTS, reason="results/ is empty until the paper run is done")
def test_make_figures_writes_exactly_the_paper_figures(tmp_path):
    pytest.importorskip("matplotlib")
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--results", str(RESULTS), "--out", str(tmp_path)]) == 0
    # The paper includes Fig. 2 (seed spread) and Fig. 3 (substitution vs MMLU); nothing else is built.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "fig_seed_spread.pdf", "fig_substitution_vs_mmlu.pdf",
    ]
