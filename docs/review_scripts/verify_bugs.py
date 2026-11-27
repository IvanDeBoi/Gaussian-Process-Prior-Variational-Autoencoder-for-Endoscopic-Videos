import torch, torch.nn as nn, time
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
dev = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. best_state aliasing: same pattern as VAEModel.train() ---
torch.manual_seed(0)
enc = nn.Linear(4, 4).to(dev)
opt = torch.optim.Adam(enc.parameters(), lr=0.1)
def state_dict_full():            # same shape as VAEModel.state_dict_full()
    return {"encoder": enc.state_dict(), "loss_total_list": []}
state = state_dict_full()
for k in list(state.keys()):
    if "list" in k: del state[k]
best_state = state                 # "snapshot" taken at the best epoch
w_at_snapshot = enc.weight.detach().clone()
for _ in range(20):                # training continues
    opt.zero_grad(); enc(torch.randn(8, 4, device=dev)).pow(2).mean().backward(); opt.step()
print("1. snapshot still equals weights at snapshot time:", torch.equal(best_state["encoder"]["weight"], w_at_snapshot))
print("   snapshot equals CURRENT weights:              ", torch.equal(best_state["encoder"]["weight"], enc.weight.detach()))

# --- 2. HPA KL: original vs rewritten ---
def hpa_orig(mu_block, s2_block, K_loc):
    b, Q, k = mu_block.shape
    L = torch.linalg.cholesky(K_loc)
    logdetK = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-12), dim=-1)
    BQ = b * Q
    mu_f = mu_block.reshape(BQ, k); s2_f = s2_block.reshape(BQ, k)
    L_e = L.unsqueeze(1).expand(b, Q, k, k).reshape(BQ, k, k)
    logdet_f = logdetK.unsqueeze(1).expand(b, Q).reshape(BQ)
    K_inv_mu = torch.cholesky_solve(mu_f.unsqueeze(-1), L_e).squeeze(-1)
    K_inv_S = torch.cholesky_solve(torch.diag_embed(s2_f), L_e)
    tr = torch.diagonal(K_inv_S, dim1=-2, dim2=-1).sum(-1)
    quad = torch.sum(mu_f * K_inv_mu, dim=-1)
    ent = torch.sum(torch.log(s2_f + 1e-12), dim=-1)
    kl = 0.5 * (tr + quad - ent + logdet_f - k)
    return kl.view(b, Q).mean() / k

def hpa_fast(mu_block, s2_block, K_loc):
    b, Q, k = mu_block.shape
    L = torch.linalg.cholesky(K_loc)                                   # (b,k,k)
    logdetK = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)   # (b,)
    Kinv = torch.cholesky_inverse(L)                                   # (b,k,k), one per batch element
    tr = torch.einsum("bk,bqk->bq", torch.diagonal(Kinv, dim1=-2, dim2=-1), s2_block)
    quad = torch.einsum("bqk,bkj,bqj->bq", mu_block, Kinv, mu_block)
    ent = torch.log(s2_block + 1e-12).sum(-1)
    kl = 0.5 * (tr + quad - ent + logdetK[:, None] - k)
    return kl.mean() / k

b, Q, k = 64, 256, 51
torch.manual_seed(1)
t = torch.sort(torch.rand(b, k, device=dev) * 60, dim=1).values
d2 = (t[:, :, None] - t[:, None, :]) ** 2
K = 2.0 / (1 + d2 / 8.0 ** 2) + 0.05 * torch.eye(k, device=dev)
K.requires_grad_(True)
mu = torch.randn(b, Q, k, device=dev); s2 = torch.rand(b, Q, k, device=dev) * 0.5 + 0.01
a = hpa_orig(mu, s2, K); c = hpa_fast(mu, s2, K)
ga, = torch.autograd.grad(a, K); gc_, = torch.autograd.grad(c, K)
print(f"2. KL orig {a.item():.6f}  fast {c.item():.6f}  rel diff {abs(a-c).item()/abs(a).item():.2e}  grad rel diff {(ga-gc_).norm().item()/ga.norm().item():.2e}")
def bench(fn, n=30):
    for _ in range(3):
        v = fn(mu, s2, K); torch.autograd.grad(v, K)
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        v = fn(mu, s2, K); torch.autograd.grad(v, K)
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3
if dev == "cuda":
    torch.cuda.reset_peak_memory_stats(); to = bench(hpa_orig); mo = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats(); tf = bench(hpa_fast); mf = torch.cuda.max_memory_allocated() / 2**20
    print(f"   forward+backward per step: orig {to:.1f} ms ({mo:.0f} MiB peak)  fast {tf:.2f} ms ({mf:.0f} MiB peak)  speed-up {to/tf:.0f}x")
else:
    print(f"   cpu: orig {bench(hpa_orig, 3):.0f} ms  fast {bench(hpa_fast, 3):.1f} ms")
