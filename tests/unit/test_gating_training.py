"""Part A: the upgraded trainer — warm-start, softmax cross-entropy, class weighting, early stopping."""

from __future__ import annotations

from slowbrain.gating_model import FEATURE_NAMES, GATING_LABELS, GatingDatasetRow, LogisticGate
from slowbrain.gating_training import _class_weights, _validation_loss, train_gate


def _row(label: str, *, feat0: float, idea: str) -> GatingDatasetRow:
    features = (feat0, *([0.0] * (len(FEATURE_NAMES) - 1)))
    return GatingDatasetRow(
        idea_id=idea,
        ticker="TIC",
        signal_date="2026-01-01",
        horizon_days=10,
        features=features,
        target_label=label,  # type: ignore[arg-type]
        baseline_label="HOLD",
        forward_return_pct=1.0,
    )


def _separable_rows() -> tuple[GatingDatasetRow, ...]:
    # feat0 = +1 -> BUY, feat0 = -1 -> HOLD: linearly separable, a linear softmax gate must learn it.
    buys = tuple(_row("BUY", feat0=1.0, idea=f"buy-{i}") for i in range(20))
    holds = tuple(_row("HOLD", feat0=-1.0, idea=f"hold-{i}") for i in range(20))
    return buys + holds


def test_training_is_deterministic() -> None:
    rows = _separable_rows()
    gate_a, _ = train_gate(rows, max_epochs=5)
    gate_b, _ = train_gate(rows, max_epochs=5)
    assert gate_a.weights == gate_b.weights


def test_softmax_cross_entropy_learns_a_separable_dataset() -> None:
    rows = _separable_rows()
    gate, _ = train_gate(rows, max_epochs=40)
    assert gate.predict_label(_row("BUY", feat0=1.0, idea="probe-buy")) == "BUY"
    assert gate.predict_label(_row("HOLD", feat0=-1.0, idea="probe-hold")) == "HOLD"


def test_warm_start_uses_the_supplied_weights() -> None:
    trained, _ = train_gate(_separable_rows(), max_epochs=20)
    # Warm-start with zero further epochs must return exactly the supplied weights...
    resumed, _ = train_gate(_separable_rows(), init_weights=trained.weights, max_epochs=0)
    assert resumed.weights == trained.weights
    # ...and a cold start with zero epochs must NOT equal the trained weights (proves it was used).
    cold, _ = train_gate(_separable_rows(), max_epochs=0)
    assert cold.weights != trained.weights


def test_warm_start_preserves_and_continues_learning() -> None:
    rows = _separable_rows()
    seed, _ = train_gate(rows, max_epochs=15)
    warm, _ = train_gate(rows, init_weights=seed.weights, max_epochs=5)
    # Resuming keeps the learned pattern (it does not reset/forget)...
    assert warm.predict_label(_row("BUY", feat0=1.0, idea="probe-buy")) == "BUY"
    assert warm.predict_label(_row("HOLD", feat0=-1.0, idea="probe-hold")) == "HOLD"
    # ...and it continued training on top of the seed (weights moved further).
    assert warm.weights != seed.weights


def test_class_weighting_favours_the_rare_label() -> None:
    rows = (
        *tuple(_row("HOLD", feat0=-1.0, idea=f"hold-{i}") for i in range(30)),
        *tuple(_row("BUY", feat0=1.0, idea=f"buy-{i}") for i in range(3)),
    )
    weights = _class_weights(rows)
    assert weights["BUY"] > weights["HOLD"]


def test_imbalanced_rare_class_is_still_learned() -> None:
    rows = (
        *tuple(_row("HOLD", feat0=-1.0, idea=f"hold-{i}") for i in range(30)),
        *tuple(_row("BUY", feat0=1.0, idea=f"buy-{i}") for i in range(4)),
    )
    gate, _ = train_gate(rows, max_epochs=60)
    assert gate.predict_label(_row("BUY", feat0=1.0, idea="probe")) == "BUY"


def test_validation_loss_drops_after_training() -> None:
    rows = _separable_rows()
    weights = _class_weights(rows)
    cold = LogisticGate(GATING_LABELS, FEATURE_NAMES, train_gate(rows, max_epochs=0)[0].weights)
    trained, _ = train_gate(rows, max_epochs=30)
    assert _validation_loss(trained.weights, rows, weights) < _validation_loss(cold.weights, rows, weights)


def test_early_stopping_runs_and_generalises() -> None:
    train_rows = _separable_rows()
    val_rows = (
        _row("BUY", feat0=1.0, idea="v-buy"),
        _row("HOLD", feat0=-1.0, idea="v-hold"),
    )
    gate, _ = train_gate(train_rows, validation_rows=val_rows, max_epochs=50, patience=2)
    assert gate.predict_label(val_rows[0]) == "BUY"
    assert gate.predict_label(val_rows[1]) == "HOLD"
    # deterministic with validation/early-stopping too
    again, _ = train_gate(train_rows, validation_rows=val_rows, max_epochs=50, patience=2)
    assert gate.weights == again.weights
