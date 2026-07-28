"""The Leduc counterfactual value network's query surface."""

import torch

from paradigm_b.core.belief import PBS, PublicState, initial_reach
from paradigm_b.leduc.value_net.features import encode_pbs
from paradigm_b.leduc.value_net.net import PBSValueNet, ValueNetConfig


def test_value_network_supports_agent_indexed_and_joint_queries():
    pbs = PBS(public=PublicState(), ranges=initial_reach())
    features = torch.as_tensor(encode_pbs(pbs)[None], dtype=torch.float32)
    net = PBSValueNet(ValueNetConfig(hidden_dim=16, num_residual_blocks=1))

    indexed = net.forward_indexed(features, torch.tensor([0]))
    joint = net(features)

    assert indexed.shape == (1, 6)
    assert joint.shape == (1, 2, 6)
    torch.testing.assert_close(indexed, joint[:, 0])
