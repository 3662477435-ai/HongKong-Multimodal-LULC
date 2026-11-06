from __future__ import annotations

import torch

from hk_landuse.hierarchical_fcnn_model import HierarchicalMultiModalFCNN


def _model(**kwargs) -> HierarchicalMultiModalFCNN:
    return HierarchicalMultiModalFCNN(
        vocab_size=8,
        s2_channels=19,
        s1_channels=0,
        urban_channels=0,
        road_channels=0,
        osm_channels=0,
        text_dim=16,
        text_heads=4,
        text_layers=1,
        max_tokens=4,
        dropout=0.0,
        use_text=False,
        **kwargs,
    ).eval()


def _inputs():
    s2 = torch.randn(1, 19, 32, 32)
    empty = torch.empty(1, 0, 32, 32)
    ids = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    return s2, empty, empty, empty, empty, ids, mask


@torch.no_grad()
def test_no_hierarchy_routes_direct_probabilities() -> None:
    result = _model(use_hierarchy=False)(*_inputs())
    assert torch.allclose(
        result["combined_log_probs"], result["direct_log_probs"], atol=1e-6
    )


@torch.no_grad()
def test_no_residual_preserves_base_logits() -> None:
    result = _model(use_semantic_residual=False)(*_inputs())
    assert torch.equal(result["direct_logits"], result["base_direct_logits"])


def test_shared_gate_has_one_trainable_scalar() -> None:
    model = _model(gate_mode="shared")
    assert tuple(model.hierarchy_gate_logit.shape) == (1,)
