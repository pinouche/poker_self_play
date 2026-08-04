"""The neural primitives both paradigms build on.

``common/nets.py`` is shared on purpose, so these tests are the contract:
if paradigm A and paradigm B ever stop agreeing on how a card is embedded
or which activation a trunk uses, one of them has drifted.
"""

import numpy as np
import torch

from common.cards import Card
from common.nets import CardEmbedding
from paradigm_a.config import Config, ModelConfig
from paradigm_a.environment.poker_env import PokerEnv
from paradigm_a.model.network import build_network, load_checkpoint, save_checkpoint
from paradigm_a.representation.observation_encoder import ObservationEncoder
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.leduc.value_net.net import PBSValueNet, ValueNetConfig

BOARD = ("As", "Kd", "7h", "2c", "Ts")


def _small_config(**model_overrides) -> Config:
    return Config(
        model=ModelConfig(hidden_dim=16, num_residual_blocks=1, **model_overrides)
    )


def _observation(cfg: Config, board=BOARD):
    """A paradigm-A observation with a known board, as a ``[1, dim]`` tensor."""
    env = PokerEnv(cfg.env, seed=0)
    env.reset()
    seat = env.state.to_act
    env.state.board = [Card.from_str(card) for card in board]
    encoder = ObservationEncoder.from_config(cfg)
    flat = encoder.encode_flat(env.get_observation(seat))
    hole = sorted(card.id for card in env.state.players[seat].hole)
    return torch.as_tensor(flat)[None], hole


def test_paradigm_a_and_b_use_gelu_and_layer_norm():
    paradigm_a = build_network(_small_config())
    paradigm_b = PBSValueNet(ValueNetConfig(hidden_dim=16, num_residual_blocks=1))

    for network in (paradigm_a, paradigm_b):
        modules = tuple(network.modules())
        assert any(isinstance(module, torch.nn.GELU) for module in modules)
        assert any(isinstance(module, torch.nn.LayerNorm) for module in modules)
        assert not any(isinstance(module, torch.nn.ReLU) for module in modules)


def test_card_embedding_sums_rank_suit_and_identity_components():
    embedding = CardEmbedding(52, 3, num_suits=4)
    with torch.no_grad():
        embedding.rank_embedding.weight.copy_(torch.arange(39).reshape(13, 3))
        embedding.suit_embedding.weight.copy_(100 + torch.arange(12).reshape(4, 3))
        embedding.card_embedding.weight.copy_(1000 + torch.arange(156).reshape(52, 3))
    cards = torch.zeros((1, 52))
    cards[0, [0, 5]] = 1.0

    encoded = embedding(cards)

    expected = (
        embedding.rank_embedding.weight[[0, 1]].sum(dim=0)
        + embedding.suit_embedding.weight[[0, 1]].sum(dim=0)
        + embedding.card_embedding.weight[[0, 5]].sum(dim=0)
    )
    torch.testing.assert_close(encoded[0], expected)


def test_both_paradigms_embed_cards_with_the_same_module():
    """The point of `common/nets.py`: one CardEmbedding class, not two."""
    paradigm_a = build_network(_small_config())
    paradigm_b = HoldemValueNet(
        HoldemValueNetConfig(hidden_dim=16, num_residual_blocks=1, card_embedding_dim=8)
    )

    for network in (paradigm_a, paradigm_b):
        assert any(isinstance(m, CardEmbedding) for m in network.modules())


def test_paradigm_a_reconstructs_exact_card_indicators_from_its_observation():
    """The observation stores rank/suit one-hots; the network needs card ids.

    Card index is ``rank * 4 + suit``, which is exactly what CardEmbedding
    factors its table on, so the outer product of the two one-hots recovers the
    card with no loss and no change to the observation layout.
    """
    cfg = _small_config()
    net = build_network(cfg)
    observation, hole = _observation(cfg)
    board = [Card.from_str(card).id for card in BOARD]

    indicators = net.card_indicators(observation)[0]
    sets = {
        name: torch.nonzero(row).flatten().tolist()
        for (name, _), row in zip(net.card_sets, indicators)
    }

    assert sets["hole"] == hole
    assert sets["flop"] == sorted(board[:3])
    assert sets["turn"] == board[3:4]
    assert sets["river"] == board[4:5]
    # Every indicator is a clean multi-hot: no fractional or doubled entries.
    assert torch.equal(indicators, indicators.round())
    assert indicators.max() <= 1.0


def test_paradigm_a_card_sets_are_bags_within_a_street_but_not_across_streets():
    cfg = _small_config()
    net = build_network(cfg)

    original, _ = _observation(cfg)
    reordered_flop, _ = _observation(cfg, ("7h", "As", "Kd", "2c", "Ts"))
    turn_swapped_in, _ = _observation(cfg, ("As", "Kd", "2c", "7h", "Ts"))

    # A hand is the same hand however you list it...
    torch.testing.assert_close(
        net.card_indicators(original), net.card_indicators(reordered_flop)
    )
    # ...but a card on the flop is not the same as that card on the turn.
    assert not torch.equal(
        net.card_indicators(original), net.card_indicators(turn_swapped_in)
    )


def test_paradigm_a_never_embeds_the_masked_opponent_slots():
    """Opponent card slots exist but are empty during play; they must stay empty."""
    cfg = _small_config()
    net = build_network(cfg)
    observation, _ = _observation(cfg)

    indicators = net.card_indicators(observation)[0]
    opponents = [
        row
        for (name, _), row in zip(net.card_sets, indicators)
        if name.startswith("opponent")
    ]

    assert opponents, "the default config keeps opponent slots"
    for row in opponents:
        assert float(row.sum()) == 0.0


def test_disabling_the_card_embedding_restores_the_flat_encoder():
    cfg = _small_config(card_embedding_dim=0)
    net = build_network(cfg)
    observation, _ = _observation(cfg)

    assert net.card_embedding is None
    assert net.card_sets == []
    logits, q_values = net(observation)
    assert logits.shape == q_values.shape == (1, net.num_actions)


def test_checkpoints_predating_the_card_embedding_load_as_the_old_architecture(tmp_path):
    """A checkpoint's config describes *its* weights, so a missing key means old."""
    cfg = _small_config(card_embedding_dim=0)
    net = build_network(cfg)
    payload = {
        "state_dict": net.state_dict(),
        "config": cfg.to_dict(),
        "observation_dim": net.spec.total_dim,
        "num_actions": net.num_actions,
    }
    del payload["config"]["model"]["card_embedding_dim"]  # as an old file would be
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)

    loaded, loaded_cfg, _ = load_checkpoint(str(path))

    assert loaded_cfg.model.card_embedding_dim == 0
    assert loaded.card_embedding is None
    # A config file the user wrote is a different matter: a key they omitted
    # means "I did not override it", so it gets the current default.
    assert Config.from_dict({}).model.card_embedding_dim > 0


def test_card_embedding_checkpoints_round_trip(tmp_path):
    cfg = _small_config()
    net = build_network(cfg)
    observation, _ = _observation(cfg)
    path = tmp_path / "net.pt"
    save_checkpoint(str(path), net, cfg)

    loaded, loaded_cfg, _ = load_checkpoint(str(path))

    assert loaded_cfg.model.card_embedding_dim == cfg.model.card_embedding_dim
    assert loaded.card_embedding is not None
    net.eval()
    with torch.no_grad():
        np.testing.assert_allclose(
            net(observation)[0].numpy(), loaded(observation)[0].numpy(), atol=1e-6
        )
