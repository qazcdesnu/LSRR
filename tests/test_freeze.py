import hashlib
import torch
import pytest
from lsrr.backbones.extractor import HFCausalBackboneExtractor
from lsrr.model import LSRRModel
from lsrr.losses.composite import CompositeLoss

def compute_model_param_hash(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

def test_backbone_strictly_frozen():
    """Test 1: Assert that backbone parameter hash is strictly unchanged after a full training step."""
    device = torch.device("cpu")
    # Load frozen extractor
    extractor = HFCausalBackboneExtractor(model_name_or_path="gpt2", device="cpu")
    initial_hash = compute_model_param_hash(extractor.model)

    # Verify requires_grad is False for every parameter
    for name, p in extractor.model.named_parameters():
        assert not p.requires_grad, f"Parameter {name} has requires_grad=True!"

    # Create LSRR model
    model = LSRRModel(
        adapter_cfg={"type": "per_layer_affine+rmsnorm", "d_model": 64},
        engine_cfg={"type": "mamba_up", "n_blocks": 1},
        fusion_cfg={"type": "attention_pooling"},
        decoder_cfg={"type": "trained_light_decoder", "n_layers": 1, "n_heads": 2},
        d_in=extractor.hidden_dim,
        num_layers=extractor.num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = CompositeLoss([{"type": "answer_nll", "w": 1.0}])

    # Dummy batch
    dummy_input_ids = torch.randint(0, 1000, (2, 8))
    with torch.no_grad():
        H = extractor.extract_hidden_states(dummy_input_ids)

    # Execute forward + backward + optimizer step on LSRR model
    optimizer.zero_grad()
    target_ids = torch.randint(0, 50257, (2, 4))
    outputs = model(H, target_ids=target_ids)
    loss_dict = loss_fn(outputs, {"target_ids": target_ids})
    loss_dict["total_loss"].backward()
    optimizer.step()

    # Verify backbone parameters remained 100% bit-exact
    after_hash = compute_model_param_hash(extractor.model)
    assert initial_hash == after_hash, f"Backbone parameter hash changed! {initial_hash} vs {after_hash}"
    print("\n[Test 1 Passed] Backbone parameters are 100% frozen and unchanged.")
