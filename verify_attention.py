"""Check the head-layout change against the upstream attention formula."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from squadgen.network.models_ae import Attention

torch.manual_seed(42)
torch.set_grad_enabled(False)
for heads, qknorm, masked in [(8, 0, False), (8, 1, True), (1, 0, True)]:
    module = Attention(64, context_dim=48, heads=heads, dim_head=16, qknorm=qknorm)
    x = torch.randn(2, 31, 64)
    context = torch.randn(2, 47, 48)
    mask = torch.rand(2, 31, 47) > 0.2 if masked else None
    q = module.to_q(x).reshape(2, 31, heads, 16).permute(0, 2, 1, 3).reshape(2*heads, 31, 16)
    k, v = module.to_kv(context).chunk(2, dim=-1)
    k = k.reshape(2, 47, heads, 16).permute(0, 2, 1, 3).reshape(2*heads, 47, 16)
    v = v.reshape(2, 47, heads, 16).permute(0, 2, 1, 3).reshape(2*heads, 47, 16)
    q, k = module.q_norm(q), module.k_norm(k)
    expanded_mask = mask[:, None].expand(2, heads, 31, 47).reshape(2*heads, 31, 47) if masked else None
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=expanded_mask, scale=module.scale)
    expected = module.to_out(expected.reshape(2, heads, 31, 16).permute(0, 2, 1, 3).reshape(2, 31, heads*16))
    actual = module(x, context=context, mask=mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    print(f'PASS: heads={heads}, qknorm={qknorm}, mask={masked}; max error {(actual-expected).abs().max().item():.3g}')
