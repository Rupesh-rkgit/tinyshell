import pytest
import torch
import torch.nn.functional as F

from tinyshell.config import ModelConfig
from tinyshell.model import TinyLM


@pytest.fixture()
def model() -> TinyLM:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        intermediate_size=64,
        max_seq_len=32,
        dropout=0.0,
    )
    return TinyLM(config).eval()


def test_forward_shape_and_loss(model: TinyLM) -> None:
    input_ids = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 8, 64)
    assert loss is not None
    assert torch.isfinite(loss)


def test_first_token_weight_changes_loss(model: TinyLM) -> None:
    input_ids = torch.randint(0, 64, (2, 8))
    targets = torch.full((2, 8), -100, dtype=torch.long)
    targets[:, 4:] = torch.randint(0, 64, (2, 4))
    _, plain = model(input_ids, targets)
    _, weighted = model(input_ids, targets, first_token_weight=5.0)
    assert plain is not None and weighted is not None
    assert torch.isfinite(plain) and torch.isfinite(weighted)
    assert not torch.isclose(plain, weighted)


def test_first_token_weight_applies_to_every_packed_example(model: TinyLM) -> None:
    torch.manual_seed(1)
    input_ids = torch.randint(0, 64, (1, 12))
    targets = torch.full((1, 12), -100, dtype=torch.long)
    targets[:, 2:5] = torch.randint(0, 64, (1, 3))
    targets[:, 8:11] = torch.randint(0, 64, (1, 3))

    logits, _ = model(input_ids)
    token_loss = F.cross_entropy(
        logits.reshape(-1, 64),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    active = (targets != -100).reshape(-1)
    weights = torch.ones(12)
    weights[[2, 8]] = 5.0
    expected = (token_loss * weights).sum() / weights[active].sum()

    _, weighted = model(input_ids, targets, first_token_weight=5.0)
    assert weighted is not None
    assert torch.allclose(weighted, expected)


def test_causal_mask_hides_future_tokens(model: TinyLM) -> None:
    input_ids = torch.randint(0, 64, (1, 12))
    changed = input_ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 64
    with torch.inference_mode():
        first, _ = model(input_ids)
        second, _ = model(changed)
    # Changing the final token must not affect logits at earlier positions.
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)
    assert not torch.allclose(first[:, -1], second[:, -1])


def test_document_mask_blocks_cross_example_attention(model: TinyLM) -> None:
    input_ids = torch.randint(0, 64, (1, 12))
    doc_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]])
    changed = input_ids.clone()
    changed[0, 2] = (changed[0, 2] + 1) % 64
    with torch.inference_mode():
        first, _ = model(input_ids, doc_ids=doc_ids, position_ids=position_ids)
        second, _ = model(changed, doc_ids=doc_ids, position_ids=position_ids)
    # The second document must not see the changed token in the first document.
    assert torch.allclose(first[:, 6:], second[:, 6:], atol=1e-6)
    # The first document's later positions do see the change.
    assert not torch.allclose(first[:, 2:6], second[:, 2:6])


def test_context_limit_is_enforced(model: TinyLM) -> None:
    with pytest.raises(ValueError):
        model(torch.zeros((1, 33), dtype=torch.long))


def test_generate_appends_tokens(model: TinyLM) -> None:
    input_ids = torch.randint(0, 64, (1, 5))
    output = model.generate(input_ids, max_new_tokens=4, temperature=0.0)
    assert output.shape == (1, 9)
    assert torch.equal(output[:, :5], input_ids)


def test_parameter_count_matches_config() -> None:
    config = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        n_heads=4,
        intermediate_size=64,
        max_seq_len=32,
        tie_embeddings=True,
    )
    model = TinyLM(config)
    # embeddings (tied with lm_head) + attention + MLP + block/final norms.
    expected = (
        128 * 32
        + 4 * 32 * 32
        + 3 * 32 * 64
        + 3 * 32
    )
    assert model.num_parameters == expected
    assert model.num_parameters_non_embedding == expected - 128 * 32
