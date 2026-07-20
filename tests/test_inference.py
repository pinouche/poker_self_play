"""The public suggest_action contract."""

import copy

import pytest

from config import Config, ModelConfig
from environment.state import ACTION_NAMES
from infer import EXAMPLE_TABLE_STATE
from inference.suggest_action import (
    SuggestionEngine,
    TableStateError,
    parse_table_state,
    suggest_action,
)


def small_config() -> Config:
    cfg = Config()
    cfg.model = ModelConfig(hidden_dim=32, num_residual_blocks=2, head_hidden=32)
    return cfg


@pytest.fixture(scope="module")
def engine() -> SuggestionEngine:
    return SuggestionEngine.untrained(small_config())


def table_state(**overrides):
    state = copy.deepcopy(EXAMPLE_TABLE_STATE)
    state.update(overrides)
    return state


# --- parsing ---------------------------------------------------------------
def test_example_table_state_parses():
    state, acting_seat, env_cfg = parse_table_state(table_state())
    assert acting_seat == 0
    assert env_cfg.big_blind == 20
    assert len(state.board) == 3
    assert state.pot == 60
    assert [str(c) for c in state.players[0].hole] == ["4h", "2s"]
    assert state.dealer == 0


def test_pot_predating_the_street_is_shared_between_live_players():
    state, _, _ = parse_table_state(table_state())
    assert sum(p.contributed for p in state.players) == 60
    assert [p.contributed for p in state.players] == [20, 20, 20]


def test_to_act_can_be_overridden():
    for name, seat in (("hero", 0), ("villain_left", 1), ("villain_right", 2)):
        state = table_state(to_act=name)
        # Give the acting player cards; only the actor needs them.
        state[name]["cards"] = [{"rank": "A", "suit": "c"}, {"rank": "K", "suit": "d"}]
        if name != "hero":
            state["hero"]["cards"] = []
        _, acting_seat, _ = parse_table_state(state)
        assert acting_seat == seat


def test_missing_player_entry_is_rejected():
    state = table_state()
    del state["villain_left"]
    with pytest.raises(TableStateError):
        parse_table_state(state)


def test_wrong_board_size_for_the_street_is_rejected():
    with pytest.raises(TableStateError):
        parse_table_state(table_state(street="turn"))  # three cards, turn expects four


def test_duplicate_cards_are_rejected():
    state = table_state()
    state["villain_left"]["cards"] = [{"rank": "4", "suit": "h"}, {"rank": "9", "suit": "c"}]
    with pytest.raises(TableStateError):
        parse_table_state(state)


def test_acting_player_without_hole_cards_is_rejected():
    state = table_state()
    state["hero"]["cards"] = []
    with pytest.raises(TableStateError):
        parse_table_state(state)


def test_inactive_acting_player_is_rejected():
    state = table_state()
    state["hero"]["active"] = False
    with pytest.raises(TableStateError):
        parse_table_state(state)


def test_pot_smaller_than_the_posted_bets_is_rejected():
    state = table_state(pot=0)
    state["hero"]["bet"] = 50
    with pytest.raises(TableStateError):
        parse_table_state(state)


def test_unknown_street_is_rejected():
    with pytest.raises(TableStateError):
        parse_table_state(table_state(street="fourth"))


# --- suggestion contract ---------------------------------------------------
def test_suggestion_has_the_documented_shape(engine):
    result = engine.suggest(table_state(), temperature=1.0)
    for key in ("action", "probabilities", "q_values", "legal_actions"):
        assert key in result
    assert set(result["probabilities"]) == set(ACTION_NAMES)
    assert set(result["q_values"]) == set(ACTION_NAMES)
    assert result["action"] in result["legal_actions"]


def test_probabilities_sum_to_one_over_legal_actions(engine):
    result = engine.suggest(table_state(), temperature=1.0)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_illegal_actions_have_zero_probability_and_null_q(engine):
    result = engine.suggest(table_state(), temperature=1.0)
    legal = set(result["legal_actions"])
    for name in ACTION_NAMES:
        if name in legal:
            assert result["q_values"][name] is not None
        else:
            assert result["probabilities"][name] == 0.0
            assert result["q_values"][name] is None


def test_unraised_flop_offers_check_and_bets_but_not_call_or_raise(engine):
    result = engine.suggest(table_state(), temperature=1.0)
    assert set(result["legal_actions"]) == {
        "CHECK",
        "BET_SMALL",
        "BET_MEDIUM",
        "BET_LARGE",
        "ALL_IN",
    }


def test_facing_a_bet_offers_fold_call_and_raises(engine):
    state = table_state(pot=120)
    state["villain_left"]["bet"] = 60
    result = engine.suggest(state, temperature=1.0)
    assert "FOLD" in result["legal_actions"]
    assert "CALL" in result["legal_actions"]
    assert "CHECK" not in result["legal_actions"]
    assert not any(name.startswith("BET_") for name in result["legal_actions"])


def test_zero_temperature_is_deterministic_and_one_hot(engine):
    first = engine.suggest(table_state(), temperature=0.0)
    second = engine.suggest(table_state(), temperature=0.0)
    assert first["action"] == second["action"]
    assert max(first["probabilities"].values()) == pytest.approx(1.0)


def test_sample_mode_returns_a_legal_action(engine):
    import random

    rng = random.Random(0)
    for _ in range(25):
        result = engine.suggest(table_state(), temperature=1.0, mode="sample", rng=rng)
        assert result["action"] in result["legal_actions"]


def test_unknown_mode_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.suggest(table_state(), mode="greedy")


def test_bet_amount_is_reported_in_chips(engine):
    result = engine.suggest(table_state(), temperature=0.0)
    if result["action"].startswith("BET_"):
        assert result["chips_to_put_in"] > 0
        assert result["bet_to"] == result["chips_to_put_in"]


def test_action_history_is_accepted_when_supplied(engine):
    state = table_state(
        action_history=[
            {"player": "hero", "action": "CALL", "amount": 20, "street": "preflop"},
            {"player": "villain_left", "action": "CALL", "amount": 10, "street": "preflop"},
            {"player": "villain_right", "action": "CHECK", "amount": 0, "street": "preflop"},
        ]
    )
    parsed, _, _ = parse_table_state(state)
    assert len(parsed.history) == 3
    assert engine.suggest(state, temperature=1.0)["action"] in ACTION_NAMES


def test_unknown_action_in_history_is_rejected():
    state = table_state(action_history=[{"player": "hero", "action": "SHOVE", "amount": 1}])
    with pytest.raises(TableStateError):
        parse_table_state(state)


def test_one_shot_helper_works_without_a_prebuilt_engine():
    result = suggest_action(table_state(), engine=SuggestionEngine.untrained(small_config()))
    assert result["action"] in result["legal_actions"]
