import torch
import torch.nn as nn
from lsrr.decoders.light_decoder import TrainedLightDecoder

def test_decoder_no_target_leakage_on_noise():
    """Verifies that the decoder cannot achieve high accuracy on random noise,
    proving that target leakage (self-copy) is eliminated by proper causal right-shift.
    """
    vocab_size = 100
    d_model = 32
    decoder = TrainedLightDecoder(d_model=d_model, vocab_size=vocab_size, n_layers=1, n_heads=2, bos_token_id=0)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(25):
        # Pure random noise memory - contains zero signal about the targets!
        h_fusion = torch.randn(8, d_model)
        target_ids = torch.randint(1, vocab_size, (8, 4))

        logits = decoder(h_fusion, target_ids)
        loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # If cheating occurred, accuracy would be near 100% and loss near 0.
    # Without cheating, accuracy on random noise must remain very low (< 25%).
    preds = logits.argmax(dim=-1)
    acc = (preds == target_ids).float().mean().item()
    assert acc < 0.25, f"Expected low accuracy on random noise, got {acc*100:.2f}% (Target leakage suspected!)"

def test_decoder_autoregressive_generate():
    """Verifies generate() produces autoregressive tokens without requiring target_ids."""
    vocab_size = 100
    d_model = 32
    decoder = TrainedLightDecoder(d_model=d_model, vocab_size=vocab_size, n_layers=1, n_heads=2, bos_token_id=0)

    h_fusion = torch.randn(4, d_model)
    gen_tokens = decoder.generate(h_fusion, max_new_tokens=6)

    assert gen_tokens.shape == (4, 6)
    assert (gen_tokens >= 0).all()
    assert (gen_tokens < vocab_size).all()
