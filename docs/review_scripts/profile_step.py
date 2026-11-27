"""Rough per-step cost breakdown with synthetic data (timing only; weights are random).
Mirrors EndoVAEEnc/EndoVAEDec and an LPIPS-VGG16-shaped perceptual loss at 256x256."""
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
dev = "cuda"; B = 16; N = 256; H = 336; D = 256
torch.manual_seed(0)

class Enc(nn.Module):
    def __init__(s):
        super().__init__(); ch = [3, 32, 64, 128, 256]; L = []
        for i in range(4): L += [nn.Conv2d(ch[i], ch[i+1], 4, 2, 1), nn.ReLU(True)]
        s.conv = nn.Sequential(*L); s.fc = nn.Linear(256*21*21, 2*D)
    def forward(s, x): return s.fc(s.conv(x).flatten(1)).chunk(2, dim=1)
class Dec(nn.Module):
    def __init__(s):
        super().__init__(); s.fc = nn.Linear(D, 256*21*21); ch = [256, 128, 64, 32, 32]; L = []
        for i in range(4): L += [nn.ConvTranspose2d(ch[i], ch[i+1], 4, 2, 1), nn.ReLU(True)]
        s.de = nn.Sequential(*L); s.out = nn.Conv2d(32, 3, 3, padding=1)
    def forward(s, z): return s.out(s.de(s.fc(z).view(-1, 256, 21, 21)))
class Percep(nn.Module):                      # LPIPS-VGG16 shaped: 5 taps, normalised feature diffs
    def __init__(s):
        super().__init__(); f = torchvision.models.vgg16(weights=None).features
        s.sl = nn.ModuleList([f[:4], f[4:9], f[9:16], f[16:23], f[23:30]])
        for p in s.parameters(): p.requires_grad = False
    def feats(s, x):
        out = []
        for blk in s.sl: x = blk(x); out.append(F.normalize(x, dim=1))
        return out
    def forward(s, a, b):
        return sum(((fa - fb) ** 2).mean() for fa, fb in zip(s.feats(a), s.feats(b)))

enc, dec, per = Enc().to(dev), Dec().to(dev), Percep().to(dev).eval()
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3)
Y_cpu = torch.randn(N, H, H, 3)                         # how the repo holds data: float32 HWC on CPU
Y_gpu_u8 = torch.randint(0, 255, (N, H, H, 3), dtype=torch.uint8, device=dev)
mean = torch.tensor([0.64, 0.36, 0.31], device=dev); std = torch.tensor([0.19, 0.16, 0.14], device=dev)
sync = torch.cuda.synchronize

def timeit(fn, n=12, warm=3):
    for _ in range(warm): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): fn()
    sync(); return (time.perf_counter() - t) / n * 1e3

def fetch_repo():
    idx = torch.randperm(N)[:B]; x = Y_cpu[idx].to(dev, non_blocking=True); return x
def fetch_gpu():
    idx = torch.randperm(N, device=dev)[:B]; return (Y_gpu_u8[idx].float() / 255 - mean) / std

def lp(recon_chw, x_chw, chunk):
    r = F.interpolate(recon_chw, size=256, mode="bilinear"); x = F.interpolate(x_chw, size=256, mode="bilinear")
    tot = 0
    for i in range(0, B, chunk): tot = tot + per(torch.tanh(r[i:i+chunk]), torch.tanh(x[i:i+chunk]).detach()) * min(chunk, B - i)
    return tot / B

def step(amp=False, cl=False, chunk=8, use_lp=True, fetch=fetch_repo):
    x = fetch().permute(0, 3, 1, 2)
    if cl: x = x.contiguous(memory_format=torch.channels_last)
    else: x = x.contiguous()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        mu, lv = enc(x); z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv.clamp(-10, 10)); rec = dec(z)
        loss = 4 * F.mse_loss(rec.float(), x)
        if use_lp: loss = loss + 2 * lp(rec, x, chunk)
    loss.backward(); opt.step()

print(f"{torch.cuda.get_device_name(0)}, batch {B}, 336x336, perceptual at 256x256")
t_fetch = timeit(lambda: (fetch_repo(), sync()))
t_fetch_gpu = timeit(lambda: (fetch_gpu(), sync()))
print(f"batch fetch   repo-style (CPU float32 gather + transfer): {t_fetch:6.1f} ms | uint8 on GPU + normalise: {t_fetch_gpu:5.2f} ms")
base_nolp = timeit(lambda: step(use_lp=False))
base = timeit(lambda: step())
print(f"full step     fp32, LPIPS chunks of 8 (as in repo):        {base:6.1f} ms   (without perceptual term: {base_nolp:.1f} ms)")
print(f"              -> perceptual term share of the step:        {100*(base-base_nolp)/base:5.0f} %")
r = {}
r["one LPIPS pass (no chunking)"] = timeit(lambda: step(chunk=B))
r["+ data as uint8 on GPU"] = timeit(lambda: step(chunk=B, fetch=fetch_gpu))
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
r["+ cudnn.benchmark + TF32"] = timeit(lambda: step(chunk=B, fetch=fetch_gpu))
enc.to(memory_format=torch.channels_last); dec.to(memory_format=torch.channels_last); per.to(memory_format=torch.channels_last)
r["+ channels_last"] = timeit(lambda: step(chunk=B, fetch=fetch_gpu, cl=True))
r["+ autocast bf16"] = timeit(lambda: step(chunk=B, fetch=fetch_gpu, cl=True, amp=True))
for k, v in r.items(): print(f"  {k:32s} {v:6.1f} ms   {base/v:4.2f}x vs repo-style")
import os; os._exit(0)
