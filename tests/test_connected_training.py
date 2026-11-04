import pytest

from lacunai.erase import connected_training as training


def rows(facts=12):
    return [
        {"fact_id": f"f{f}", "prompt": f"p{f}-{i}", "answer": "42"}
        for f in range(facts)
        for i in range(8)
    ]


UTILITY = [{"prompt": "general utility", "answer": "oak"}]


def test_balanced_schedule_and_validation():
    data = rows()
    schedule = training.validate_rows(data, UTILITY, 24)
    assert [data[i]["fact_id"] for i in schedule] == [f"f{i}" for i in range(12)] * 2
    for invalid in (True, 0, 513):
        with pytest.raises(ValueError):
            training.validate_rows(data, UTILITY, invalid)
    for broken in (rows(9), data[:-1]):
        with pytest.raises(ValueError, match="eight examples"):
            training.validate_rows(broken, UTILITY, 1)
    with pytest.raises(ValueError, match="overlapping"):
        training.validate_rows(data, [{"prompt": data[0]["prompt"], "answer": "oak"}], 1)
    assert len(training.validate_rows(rows(8), UTILITY, 512)) == 512


def test_projection_and_independent_storage_bound():
    torch = pytest.importorskip("torch")
    original = torch.ones(4, 4)
    current = original * 2
    result = training.project(current, original)
    assert result["projection_scale"] == pytest.approx(0.24)
    assert result["post_projection_relative"] == pytest.approx(0.24, abs=1e-7)
    with pytest.raises(ValueError, match="relative bound"):
        training._relative(original * 1.26, original)
    with pytest.raises(ValueError, match="original norm"):
        training.project(current, torch.zeros_like(original))


def test_actual_training_callback_and_restore(monkeypatch):
    from tests.test_factoid_gradient import toy

    torch, runtime, name, events = toy(monkeypatch)
    before = training._hashes(runtime.model)
    flags = [p.requires_grad for p in runtime.model.parameters()]
    seen = []

    def callback(step, replacements, diagnostics):
        assert torch.equal(dict(runtime.model.named_parameters())[name], replacements[name])
        assert diagnostics["trace"][-1]["step"] == step
        seen.append(step)
        return False

    result = training.train_connected(
        runtime, rows(), UTILITY, steps=2, checkpoint_callback=callback
    )
    assert seen == [2]
    assert training._hashes(runtime.model) == before
    assert [p.requires_grad for p in runtime.model.parameters()] == flags
    assert all(p.grad is None for p in runtime.model.parameters())
    assert result["diagnostics"]["completed_steps"] == 2
    assert not torch.equal(
        result["replacements"][name], dict(runtime.model.named_parameters())[name]
    )
    assert events[0]["state"] == "active" and events[-1]["state"] == "complete"

    def fail(*args):
        raise RuntimeError("injected callback failure")

    with pytest.raises(RuntimeError, match="injected"):
        training.train_connected(runtime, rows(8), UTILITY, steps=1, checkpoint_callback=fail)
    assert training._hashes(runtime.model) == before
    assert events[-1]["state"] == "failed"


def test_rounded_bound_guard_and_restore(monkeypatch):
    from tests.test_factoid_gradient import toy

    torch, runtime, name, _ = toy(monkeypatch)
    before = training._hashes(runtime.model)
    original_round = training.round_to_storage
    calls = 0

    def bad_round(values, dtypes):
        nonlocal calls
        calls += 1
        result = original_round(values, dtypes)
        if calls > 1:
            result[name] = result[name] * 2
        return result

    monkeypatch.setattr(training, "round_to_storage", bad_round)
    with pytest.raises(ValueError, match="relative bound"):
        training.train_connected(runtime, rows(), UTILITY, steps=1)
    assert training._hashes(runtime.model) == before


def test_bfloat16_proposal_and_callback_precision(monkeypatch):
    from tests.test_factoid_gradient import toy

    torch, runtime, name, _ = toy(monkeypatch)
    parameter = dict(runtime.model.named_parameters())[name]
    with torch.no_grad():
        parameter.copy_(parameter.bfloat16().float())
    runtime.storage_dtypes[name] = "BF16"
    before = training._hashes(runtime.model)

    def callback(step, replacements, diagnostic):
        value = replacements[name]
        assert torch.equal(value, value.bfloat16().float())
        assert torch.equal(parameter, value)
        runtime.model.train()
        for p in runtime.model.parameters():
            p.requires_grad_(False)
        return False

    result = training.train_connected(
        runtime, rows(), UTILITY, steps=1, checkpoint_callback=callback
    )
    assert training._hashes(runtime.model) == before
    value = result["replacements"][name]
    assert torch.equal(value, value.bfloat16().float())
    assert result["diagnostics"]["trace"][0]["rounded_relative"] <= 0.25


@pytest.mark.parametrize("facts", [4, 5])
def test_small_profile_explicit_and_balanced(facts):
    data = rows(facts)
    schedule = training.validate_rows(data, UTILITY, facts * 2, profile="connected-5")
    assert [data[i]["fact_id"] for i in schedule] == [f"f{i}" for i in range(facts)] * 2
    with pytest.raises(ValueError, match="profile fact groups"):
        training.validate_rows(data, UTILITY, 1)
    for legacy in (8, 12):
        with pytest.raises(ValueError, match="profile fact groups"):
            training.validate_rows(rows(legacy), UTILITY, 1, profile="connected-5")


@pytest.mark.parametrize("profile", [None, True, "unknown", [], {}])
def test_unknown_training_profile_fails_before_torch(profile):
    with pytest.raises(ValueError, match="unknown connected training profile"):
        training.validate_rows(rows(5), UTILITY, 1, profile=profile)


def test_small_actual_proposal_and_restore(monkeypatch):
    from tests.test_factoid_gradient import toy

    torch, runtime, name, _ = toy(monkeypatch)
    before = training._hashes(runtime.model)
    result = training.train_connected(runtime, rows(5), UTILITY, steps=1, profile="connected-5")
    assert result["diagnostics"]["profile"] == "connected-5"
    assert training._hashes(runtime.model) == before
    assert not torch.equal(
        result["replacements"][name], dict(runtime.model.named_parameters())[name]
    )


@pytest.mark.parametrize(
    "profile,facts,expected",
    [
        ("connected-5", 5, [16, 17]),
        ("connected-12", 12, [17]),
    ],
)
def test_profile_bound_actual_callback_cadence(monkeypatch, profile, facts, expected):
    from tests.test_factoid_gradient import toy

    _, runtime, _, _ = toy(monkeypatch)
    callbacks = []

    def callback(step, replacements, diagnostics):
        callbacks.append(step)
        return False

    result = training.train_connected(
        runtime, rows(facts), UTILITY, steps=17, profile=profile, checkpoint_callback=callback
    )
    assert callbacks == expected
    assert result["diagnostics"]["development_checks"] == len(expected)
    assert result["diagnostics"]["development_check_every"] == (16 if facts == 5 else 64)
    assert result["diagnostics"]["maximum_development_checks"] == len(expected)
    assert result["diagnostics"]["completed_steps"] == 17


def test_profile_screen_budgets_preserve_optimizer_cap():
    assert training.MAX_STEPS == 512
    assert training.PROFILE_CHECK_EVERY == {"connected-12": 64, "connected-5": 16}
    assert {
        key: training.MAX_STEPS // cadence for key, cadence in training.PROFILE_CHECK_EVERY.items()
    } == {
        "connected-12": 8,
        "connected-5": 32,
    }
