# GP-VAE Endo: review findings

Findings from a read-only review of the code (then in `15 GPVAE Endo/`, now this repo's root) on 2026-11-10;
B1–B13 were found before any code changed. Later findings (B14 on) come from work on branch `rewrite`.
Line numbers refer to commit `aaf8b6d`. IDs match the improvement plan.

- Field guide (how the code works, HPC runbook): https://claude.ai/artifact/MeRZLvfqkkUZvHi4ho3Tia
- Improvement plan (fixes, speed, structure, roadmap): https://claude.ai/artifact/MAykNBRKdt4jHHbq5cvAs5
- Test scripts: `docs/review_scripts/verify_bugs.py`, `docs/review_scripts/profile_step.py` (need only PyTorch
  and torchvision; synthetic data, no dataset).

"Tested" = reproduced with a script. "Read" = established from the source alone.

## Status (branch `legacy-refactored`)

This branch restructures the original code (commit `aaf8b6d`) into the `gpvae_endo` package without
changing what it computes. B1 and B2 are fixed on purpose (both change results); B10 and the debug
slow-downs are fixed without changing any number. The entries below describe the original code; where
they say "fixed in phase 1/2" they mean branch `rewrite`, which is a separate, experimental line of work.

| ID | Status on this branch |
|---|---|
| B1 | **Fixed** (`gpvae_endo/evaluation/metrics.py`, `Denormaliser`; `tests/test_metrics.py`). Every metric changes; the only other output it touches is the (off by default) 2-D latent thumbnail plot |
| B14 | Gone as a side effect of the restructure (the PSNR helper is now a module-level function); only runs with `MUST_USE_MASK = False` reached it, and they crashed before |
| B2 | **Fixed** (2026-11-19, after the first commit): the best-epoch snapshot is a real CPU copy, taken only among epochs with the full KL weight (`beta >= max_beta`); earlier epochs have a lower loss only because the KL weight is lower. MSVQ (no KL) chooses among all epochs; on a GPU it already copied, so its GPU results are unchanged. A run that ends before the full KL weight keeps its last epoch and still calls the restore, so B15's random draws happen as before. `tests/test_equivalence.py`: training identical to the original; final weights = best eligible epoch; when that is the last epoch (all tiny cases) the whole fingerprint equals the original; a forced case checks a non-last epoch is restored |
| B10 | **Fixed**: `gpvae_endo/utils/device.py`; without CUDA the run stops (exit code 1) unless `GPVAE_ALLOW_CPU=1` (`tests/test_device.py`) |
| Debug slow-downs | **Removed**, numbers unchanged: `CUDA_LAUNCH_BLOCKING=1` / `TORCH_USE_CUDA_DSA=1`, 12 `.item()` GPU syncs per step (loss sums now float64 tensors on the device: same IEEE double arithmetic, one read-back per epoch), `torch.cuda.empty_cache()` after each best-epoch snapshot |
| B8 | Kept: the train FID is still computed on the same frames, but the duplicate second computation and print were dropped. The throw-away FID object is still built, so the random state is unchanged |
| B3, B4, B5, B7, B9, B11, B12, B15, B16, B17, B20, B21, B22, B23, B24 | Kept on purpose (they change results; out of scope for a behaviour-preserving refactor). `tests/test_equivalence.py` shows the refactored training matches the original bit for bit; B9 has its own test |
| B6, B18, B19 | Kept: they live in the data preparation scripts (now `scripts/`), which were only moved |
| B13 | B13's dead code (`SVGPVAE_Jazbec_model.py`, the duplicate FID) was removed |
| T2 | Applied: the faster HPA KL formula (`tests/test_hpa_kl.py` compares it with the original code) |
| Checkpoints | Added (`--checkpoint`, exit code 85, `SLURM_JOB_END_TIME`); a resumed run equals a straight run bit for bit (`tests/test_checkpoint.py`) |

## Bugs

### B1. Post-training metrics are computed on a 3-pixel strip (tested)
`denormalise()` (`Latent_Space_Discovery.py:1043`) writes `arr[:, :, c]`. For one image (H, W, C) that is the
colour channel; for the (N, H, W, C) batches used by RMSE, PSNR, SSIM and FID it is the width axis, so only pixel
columns 0–2 are filled and the rest stay 0. Evidence: on a (2, 336, 336, 3) array 99.1% of the output is 0; every
row of `Experiment_Summaries.txt` has SSIM ≈ 1.0000 and PSNR 50–60 dB; the paper's Table 3 shows the same signature
(SSIM 0.99997, PSNR ≈ 55 dB). Training, saved models and the per-frame PNG/MP4 reconstructions are not affected
(they call `denormalise()` per image). Fix: index with `[..., c]`.

**Measured effect (phase 1).** One saved model (the golden VAE: 5 epochs, `c1_descending_t1_v2`) was evaluated
twice, once by the old code and once by the fixed code. Only the metric code differs.

| Metric (train / test) | Old `denormalise` | Fixed |
|---|---|---|
| RMSE | 0.0243 / 0.0242 | 0.197 / 0.196 |
| SSIM | 0.996 / 0.996 | 0.222 / 0.223 |
| PSNR (dB) | 32.30 / 32.31 | 14.13 / 14.13 |
| FID | 13.2 / 13.1 | 443 / 429 |

A 5-epoch model is barely trained, so the fixed numbers are poor, as they should be. The old code reported
near-perfect SSIM for it. Every metric in `Experiment_Summaries.txt` and in the paper's Table 3 that came from
this code path has to be recomputed.

### B2. The "best" epoch is never restored (tested)
`VAEModel.train()` keeps `best_state = self.state_dict_full()` (`Models/VAE_model.py:946–952`). `state_dict()`
returns the live tensors, not copies, and the optimiser updates them in place, so the restore at line 1015 loads
the model into itself. Evidence: `verify_bugs.py`, same pattern, 20 further updates: snapshot == current weights,
!= weights at snapshot time. VAE, HPA and SPA save the last epoch; MSVQ-VAE copies with `.cpu()` and saves its best
epoch. Fix: `copy.deepcopy`, track the best only after the β ramp, ideally on held-out frames.

### B3. LoRA learning rate differs by 100× between VAE and GP models (read)
`GPVAECasaleModel` rebuilds the optimiser with `get_trainable_encoder_params(lr)` (`GPVAE_Casale_model.py:92`), so
`lora_lr` falls back to the base rate 1e-3 for HPA and SPA, while the VAE uses 1e-5 (the paper states 1e-5).
Fix: pass `enc_lr=self.encoder_head_lr, lora_lr=self.lora_lr`.

### B4. In GPVAE-HPA the KL term never reaches the encoder (tested)
`GPVAEHPAModel.compute_kl_term` ignores the batch's `mu`/`logvar` and uses `mu_all`/`logvar_all`, refreshed under
`torch.no_grad()` once per epoch (`GPVAE_HPA_model.py:136–262`). Its gradient reaches only the GP lengthscale,
scale and noise. The HPA encoder is trained by reconstruction alone, and its log σ² is free to collapse. Appendix
A.2 of the paper mentions the caching and says SPA still regularises the encoder; it says nothing of the kind for
HPA. Tested 2026-11-19 (branch `legacy-refactored`, tiny 32-px models, after `refresh_epoch_state()`): backpropagating the KL of one batch gives an encoder gradient of exactly 0 for HPA, 1,991 (sum of |grad|) for SPA; both reach the GP lengthscale, scale and noise. Still to check on real runs: mean `logvar_all` of HPA vs VAE after training. Fix: use live μ, σ for the batch row of each
block and cached values for the neighbours.

### B5. The KL weight is not comparable across models (read, computed)
Per latent dimension and frame: VAE sums over 256 dims with β → 0.01 (weight 2.56); HPA averages over dims and
divides by the block size with β → 0.001 (0.001); SPA averages and multiplies by N/B (0.005 at 330 training frames,
0.025 at 1,600). That is ~2,560× and 100–500× less pressure than the VAE, and SPA's grows with video length.
Together with B3 and B4, VAE vs GP-VAE is not a like-for-like comparison: lower reconstruction error of the GP
models does not yet show that the temporal prior helps. Control: a VAE with β = 0; one KL unit and β schedule for
all models; 3–5 seeds.

### B6. Frame extraction mislabels frame numbers (read)
`Utils/Split_mp4_in_images.py:27–41` names files from `start_index` but reads the video from frame 0, so
`…_001000.png` is video frame 0. The GP only sees a constant shift in x, but any join with dataset annotations by
frame number is off by `start_index`. Fix: decode sequentially and name files by the true index.

### B7. The perceptual loss sees distorted colours and ignores the mask (read)
`_compute_lpips_chunked` passes `torch.tanh` of GastroNet-standardised pixels to LPIPS (`VAE_model.py:566–568`,
`MSVQ_VAE_model.py:626`); LPIPS expects RGB in [−1, 1]. tanh shifts the colour balance per channel and flattens
values beyond ~2 standard deviations. The specular mask is applied to MSE only. Fix: undo the standardisation,
`(x·std + mean)·2 − 1`; optionally average LPIPS's spatial map over valid pixels.

### B8. FID on too few frames, and computed twice (read)
FID uses 2,048-d Inception features on 208–617 frames, so the covariance is rank-deficient. The train FID block is
duplicated (`Latent_Space_Discovery.py:2813–2861`). Fix: remove the duplicate; use masked LPIPS or KID instead.

### B9. The exact-GP model crashes when the KL switches on (read)
`GPVAECasaleModel.compute_kl_term` (`GPVAE_Casale_model.py:167`) does not accept `Xb_norm`/`batch_idx`, which
`compute_total_loss` passes, so `GPVAE_Casale` raises a TypeError at epoch 51, the first with β > 0.

### B10. Silent CPU fallback (read)
`config_endo.py:254` and `VAE_model.py:115` fall back to CPU when no GPU is visible, without an error. On a cluster
that burns the whole wall time. Fix: assert a GPU at start.

### B11. Model selection and early stopping use the training loss (read)
`VAE_model.py:946, 1009` watch the training total, whose KL weight changes during the ramp. Held-out frames are
never used during training.

### B12. Two different uncertainties shown under one label (read)
`reconstruct_and_save_all_frames` (`Latent_Space_Discovery.py:1457–1488`) shows encoder σ for training frames and
GP posterior σ for held-out frames as the same quantity.

### B13. Dead and duplicated code (read)
`LoRALinear.forward` and `ResidualBlock` are each defined twice (`Encoder_Decoder.py:695/747`, `638/912`);
`_to_numpy` five times in the driver; the GP hyper-parameter priors in `GP.py` never enter a loss; `DuckNet` loads
with `strict=False`; `SVGPVAE_Jazbec_model.py` and `Trash/` are unused; `requirements.txt` is UTF-16 with Windows
conda paths.

### B14. Unmasked PSNR raises NameError (tested)
`compute_psnr_from_denorm` (`Latent_Space_Discovery.py:2076–2081`) calls `arr_to_chw_float01`, which is a local
function of `compute_ssim_from_denorm` (line 2046), so the `M is None` branch raises `NameError`. Only runs with
`MUST_USE_MASK = False` reach it; the default config masks, so past results are unaffected. Evidence:
`tests/test_metrics.py::test_psnr_unmasked_path` (xfail, `raises=NameError`). Fix: move the helper to module level.

### B15. A GP model's optimiser is detached from its GP after loading a state (tested)
`load_state_dict_full` in the GP models builds a **new** `GP` object (`GPVAE_HPA_model.py:299`,
`GPVAE_Casale_model.py:294`). The optimiser created in `GPVAECasaleModel.__init__` (lines 76–98) keeps the
old lengthscale, outputscale and noise tensors, so any training after a load updates tensors the model no
longer uses: the GP hyper-parameters freeze without an error. `train()` itself ends with
`load_state_dict_full(best_state)`, so this is the state of every trained GP model; it also draws three more
numbers from numpy's global RNG (`GP.py:102–108`). One training run followed by evaluation is not affected.
Continuing training on the same object, or a naive checkpoint resume, is. Evidence:
`tests/test_checkpoint.py::test_legacy_load_state_dict_full_orphans_the_optimiser`, and a negative control
in the same file where a resume through the legacy loader leaves ℓ, s and noise identical over epochs 4–7.
The phase 2 resume path loads into the existing `gp_prior` instead; the legacy loader is unchanged.

### B16. `ProgressTracker` ignores `print_iterations` (read)
`Progress_Tracker.py:15` sets `self.print_iterations = 1` whatever is passed. The effect is one log line per
epoch (harmless), and the "every 1000 × print_iterations epochs" latent plot in `train()` fires every 1,000
epochs.

### B17. A non-finite loss is skipped in the VAE but stops an HPA run (read)
`VAEModel.train()` skips a step whose total loss is not finite (`VAE_model.py`, the `NaN total loss` branch),
but `GPVAEHPAModel.compute_kl_term` raises `ValueError("Non-finite HPA KL encountered")`
(`GPVAE_HPA_model.py:259`) before that check is reached. With the EndoVAEEnc instability (P3), a spike after
the warm-up can therefore end an HPA job. SPA had the same raise. Fixed in phase 2: both raises are removed,
so the step is skipped as in the VAE.

### B18. DUCKNet runs without its normalisation layers, so the "specular mask" is a brightness threshold (tested)
The checkpoint `Duck16mod_Spec_99_new.pt` holds 1,410 BatchNorm tensors. `Utils/MaskPixels.py:326–340` builds
`DuckNet(normalization=None)` and loads with `strict=False`, which drops them without a message. Probe on 5
Kvasir, 3 Hangar and 2 C3VD frames: the network as used fires (p > 0.1) on about 3.1% of its 256² input for
every image, and only inside the reflected-padding border that is cropped away, so on **0 in-frame pixels**.
With the BatchNorm layers restored (a strict load then works) it still fires only in the border on these
frames. The mask therefore comes from the fallback "grey value > 225, dilated twice": the C3VD golden NPZ is
71.4% valid on average (minimum 57.2%), and Kvasir frames 1000–1015 of `8a00709108cd4e2b` are 100% valid.
Effect: masked MSE and all masked metrics exclude bright pixels, not detected specular highlights; on Kvasir
the mask currently does nothing. `gpvae_endo/masks.py` reproduces the legacy behaviour exactly (tested against
`detect_specular_ducknet`). Whether to fix the network loading is a question for Ivan: it changes every mask
and with it the training data.

### B19. The legacy mask function reloads the 271 MB DUCKNet checkpoint for every frame (tested)
`detect_specular_ducknet` builds the model and calls `torch.load` on each call: about 0.8 s per frame on the
CPU against 0.3 s when loaded once (`gpvae_endo/masks.SpecularMasker`, identical masks).

### B20. FID input is truncated, not rounded (tested)
The FID helper converts with `(x*255).astype(uint8)` (`Latent_Space_Discovery.py:2830–2832`). On 40 golden
frames 13.8% of the *real* frames' values come out one level below the original uint8. Together with B8
(rank-deficient covariance) the FID values carry little meaning.

### B21. Settings read from two places can disagree (read)
The driver scales x with the `--config` file's `USE_X_SCALING`, while `GP.py:5/34` chooses the lengthscale
bounds from `config_endo.USE_X_SCALING`; a config that sets it to True would scale x to [0, 1] under bounds of
[1, 30] frames. The same split applies to warm-up, ramp, loss weights and LPIPS size (P2).
`gpvae_endo.train` prints a warning listing every such value at start-up.

### B22. An epoch in which every step was skipped counts as the best epoch (tested)
When every step of an epoch is non-finite and skipped, `train()` reports an epoch total of 0.0. Once β has
reached its maximum, that 0.0 is taken as the best loss (a best-state snapshot is made) and as an improvement
for early stopping. Evidence: `tests/test_stability.py` (NaN in `mu_all` for one epoch). Open; pre-existing.

### B23. The GP mean constant is trainable but in no optimiser (read)
`gp_prior.mean_module.raw_constant` requires grad and is never given to the optimiser. Harmless: the GP KL
terms use a zero mean.

### B24. The exact-GP KL computes the wrong trace term (tested)
`GPVAECasaleModel.gp_prior_kl_full` (`GPVAE_Casale_model.py:236–238`) is meant to compute tr(K⁻¹Σ) (the comment
says so), but `W = cholesky_solve(Σ^½, L)` is K⁻¹Σ^½, not L⁻¹Σ^½, so `sum(W * W)` is tr(Σ^½K⁻²Σ^½) =
tr(K⁻²Σ). On a random 6×6 SPD K: original code 0.0496 = tr(K⁻²Σ), correct tr(K⁻¹Σ) 0.426. Fix: a triangular
solve with L, or `diag(K⁻¹)·σ²` as in the HPA rewrite (T2). No effect on any past run, because B9 crashes
Casale before this function is ever reached. The function is also slow: a Python loop over the Q latent
dimensions, each with an N×N right-hand side (O(Q·N³) time, N² memory per step); one K⁻¹ shared by all
dimensions would be about Q× faster. Both kept on branch `legacy-refactored` (unreachable while B9 stands).

## Properties of the method as implemented

### P1. The GP posterior uncertainty does not depend on frame content
`time_encode()` (`GPVAE_Casale_model.py:376–452`) returns std = sqrt(diag(Kss − Kxsᵀ Kxx⁻¹ Kxs)): it depends only
on frame numbers and kernel settings and is identical for all 256 latent dimensions. On a video with no missing
frames it is nearly constant.

### P2. Whole videos are held in RAM, and nothing is saved until training ends
Frames are float32 (1.35 MB per frame per copy, several copies during evaluation); the model is written to disk
only after `train()` returns (`Latent_Space_Discovery.py:2449–2456`). The C3VD videos in the paper have 208–617
frames. Settings are split between the `--config` file and `config_endo.py`, which four model files import directly.

### P3. EndoVAEEnc training blows up regularly (tested on past logs and a step-level rerun)
In Ivan's `Logs/`, the per-epoch training total jumps more than 5× (and above 50) at least once in
**28 of 41** EndoVAEEnc runs of at least 50 epochs. The largest epoch total is 1.3e31. That happens in
0 of 43 GastroEnc runs, 0 of 13 CNNEnc runs, 0 of 5 MSVQ runs and 1 of 6 GaLoRaEnc runs. The runs usually
recover within a few epochs.

With the old `tanh` LPIPS input, the 5-epoch golden run happened not to hit a spike. After the B7 fix it
hits one at step 10 (epoch 3). There the gradient norm before clipping goes 45 → 184,000, MSE goes
5.2 → 289, and the run recovers by step 13. `docs/review_scripts/diagnose_b7.py` logs every step. Clamping
the LPIPS input to [−1, 1] gives the identical blow-up, and LPIPS inputs were within [−0.76, 1.19] the step
before. So the LPIPS input does not cause it; the B7 fix only moved the trajectory onto a spike.

Likely cause (not yet tested): EndoVAEEnc's single Linear(112,896 → 512) and the decoder's
Linear(256 → 112,896) with Adam at lr 1e-3. Adam moves every weight by about the learning rate whatever the
gradient size, so clipping the gradient norm at 1.0 does not bound the update. With 112,896 inputs summed
coherently, one update can shift a unit's output by a lot. The bug B2 meant that a late spike was always
what got saved. This affects every EndoVAEEnc result. Remedies to test are listed as M11 in the plan.

### P4. The GP learning rate decays 1,000×, the network's 100× (read, and seen in the LR trace)
The GP parameter groups start at 10× the base rate (1e-2), but the cosine schedule's `eta_min = 1e-5` is one
value for all groups. By the last epoch the GP rate has fallen to 1e-5 like the network's. Recorded per epoch
in `tests/tiny_reference.json`.

### P5. The golden run exercises little of the training code (read)
The golden run stops after 5 epochs, inside the 50-epoch warm-up. It never computes a KL with β > 0, never
enters the cosine schedule, never takes a best-state snapshot and never reaches early stopping, and it is a
VAE, so no GP code runs. It also sits on a P3 spike, which widens its tolerance to 3.3%. Since phase 2,
`tests/test_characterisation.py` covers those paths for VAE, HPA and SPA on tiny synthetic data on the CPU
(warm-up 2, ramp 2, 8 epochs), with a recorded fingerprint in `tests/tiny_reference.json`.

### P6. Masked PSNR is −20·log10(masked RMSE); it is not comparable with the unmasked PSNR (tested)
With a mask, PSNR and RMSE both come from the masked MSE of the *unclipped* denormalised frames (golden
reference: −20·log10(0.200165) = 13.972 = PSNR_tr_im). SSIM and the unmasked PSNR clip to [0, 1] first.

### P7. The test frames are the same for every seed, and only the first model of a driver run starts from the seed (read)
The random split uses a hard-coded `random_state=42`. In a driver run over several `MODELS`, later models
inherit the RNG state the previous training left. `gpvae_endo.train` trains one model per process and always
starts from the seed, so it matches the driver only for the first model of a run.

### P8. HPA's image path behaves like a plain autoencoder: it fits the training frames far better than the test frames (measured; cause read)
First full runs (T6), 5 seeds each, masked metrics on the whole frame:

| | PSNR train (im) | PSNR test (im) | gap | SSIM test (im) |
|---|---|---|---|---|
| VAE | 33.55 ± 0.02 | 30.75 ± 0.03 | 2.8 dB | 0.883 ± 0.001 |
| GPVAE_HPA | 41.29 ± 0.07 | 32.20 ± 0.08 | 9.1 dB | 0.865 ± 0.001 |

HPA beats the VAE on test PSNR/RMSE by 1.45 dB (≈ 20 standard deviations), but loses on test SSIM, and its
train–test gap is 3× the VAE's. The `_im` path encodes the frame and decodes it; it does not use the GP. By B4
the HPA KL sends no gradient to the encoder or decoder, so HPA's networks are trained by 4·MSE + 2·LPIPS
alone, i.e. as a plain autoencoder, while the VAE's carry a KL of weight 2.56 per dimension (B5). The result
fits that: tighter fit to training frames, weaker generalisation of structure. **So the HPA-vs-VAE `te_im`
difference measures "no KL vs KL", not the temporal prior.** Not yet checked: HPA's `logvar_all` in
`latents.npz` (collapsed towards the −10 clamp would confirm); a VAE with β = 0 as the control (M1).

### P9. Reading a config file re-seeds the random generators (tested)
`config_endo.py` and every `configs/*.py` end with `np.random.seed(SEED)` and `torch.manual_seed(SEED)`
(`config_endo.py:267–268`). The model files import `config_endo`, so in the original the seed is applied
at import time, again by the driver, and once more whenever a config is executed. Effect: anything that
reads a config file *after* the run seeded its generators restarts the random sequence and changes every
result. Evidence: the first refactored equivalence run differed from epoch 0 until the settings were read
before seeding; the pipeline and the test harness now read all config files first (as the original's imports
did). Note for future work: config files should not seed.

### P10. MSVQ-VAE trains differently from the other models in three hidden ways (read)
- It ignores `LOSS_WEIGHTS` and uses its own fixed weights {mse 5, lpips 2, gan 0.01, kl 1}
  (`MSVQ_VAE_model.py`; now `MSVQ_LOSS_WEIGHTS` in `gpvae_endo/models/msvq_vae.py`).
- `model.train()` is `nn.Module.train(True)` followed by training, so the LPIPS network it owns is switched
  to training mode: its dropout layers are active in the perceptual loss of MSVQ, but not of the VAE/GP models.
- The frame order is shuffled once before the first epoch and then reused every epoch.
All three are kept. They matter when MSVQ is compared with the other models.

## Measurements (laptop RTX 4070, synthetic frames; proportions only)

### T1. Where a training step spends its time
`profile_step.py`, batch 16, EndoVAE layer shapes, VGG16-shaped perceptual loss at 256 px: 428 ms per step, 103 ms
without the perceptual term (76% of the step). bf16 autocast: 420 → 274 ms (1.5×).

| LPIPS resolution | float32 | bf16 autocast |
|---|---|---|
| 256 px | 420 ms | 274 ms |
| 192 px | 276 ms | 181 ms |
| 128 px | 174 ms | 118 ms |
| 64 px | 114 ms | 81 ms |

No measurable effect (within 2%): unchunked LPIPS, uint8 frames on the GPU (batch fetch 5.7 → 1.1 ms), cuDNN
benchmark + TF32, channels-last.

### T2. The HPA KL can be computed exactly without the per-dimension expansion
Using diag(K⁻¹) for the trace term and one contraction for the quadratic term instead of copying the Cholesky
factor per latent dimension: identical value, gradient within 5e-7, 58 → 3.6 ms and 1,333 → 34 MiB per step at
b=64, Q=256, k=51 (`verify_bugs.py`).

### T3. Batch 64 is very slow on the 8 GB laptop GPU
The golden run (VAE + EndoVAEEnc, 166 training frames, batch 64, LPIPS at 256 px) trained 5 epochs in
8 min 38 s, about 100 s per epoch for three batches. `profile_step.py` predicts about 1.7 s per batch of 64.
There was no out-of-memory error. The likely cause is Windows spilling GPU memory into shared system memory;
this has not been verified. A100 timings are not affected. For laptop runs, use a smaller batch.
Confirmed in phase 2: `status.json` of the golden configuration reports a peak GPU allocation of 14.5 GB
(`torch.cuda.max_memory_allocated`) on the 8 GB card, at 100 s per epoch.

### T4. Tiny CPU training runs are bit-reproducible at a fixed thread count
24 synthetic frames of 32×32, EndoVAEEnc/Dec, LPIPS at 16 px, 8 epochs, `torch.set_num_threads(1)`: two
separate processes write byte-identical fingerprints (losses, learning rates, weight norms, GP parameters).
With 4 threads the run is about 3× faster but the numbers differ from the 1-thread ones, so the tests fix the
thread count. This is what lets the resume tests demand exact equality.

### T5. The HPA KL rewrite: same value, better gradients, 66× faster on the CPU
`hpa_kl_batch` now uses diag(K⁻¹) and one contraction (T2). Against the old code kept in
`tests/test_hpa_kl.py`, float32, Q = 256, integer frame numbers, k = 51 and 4, lengthscale 1 and 30, noise
1e-3 and 1, log-variance −10 and +10:
- values differ by at most 1.1e-7 relative;
- gradients differ by at most 3e-6 in most settings; where they differ more, the new form is closer to a
  float64 reference (outputscale gradient error at ℓ = 30: 3.2e-6 new, 1.5e-4 old; with noise 1e-3: 3.0e-4
  against 3.4e-3);
- CPU, b = 64, Q = 256, k = 51: 1,278 ms → 19.4 ms per call, peak 1,311 MiB → 6 MiB.
In the tiny HPA training run only the printed KL moved (at most 1.1e-7 relative); losses, weights and GP
parameters are unchanged to the last bit.

### T6. First full runs on CalcUA (A100, Kvasir clip): timing, stability, GP restoration
Clip `kvasir_1000-2999` (frames 1000–2999 of Kvasir-Capsule `8a00709108cd4e2b`, 336², valid pixels 0.9989),
EndoVAEEnc/Dec, config defaults (500 epochs, batch 64, warm-up 50, ramp 50), split 1,600 train / 400 random
test (P7), seeds 1–5, code `fdb9498`, one A100-SXM4-40GB per job. Collected with `slurm/summarize_runs.py`
(2026-11-19).
- **Timing**: 9.1–10.7 s/epoch for both models; the HPA KL (on from epoch 50) adds nothing visible
  (warm-up-only test: 10.7 s/epoch). 500 epochs ≈ 1.5 h, one job each, no resubmission. Peak GPU memory
  14.5 GB (3-epoch test).
- **Stability (P3)**: 0 skipped steps in all 10 runs. Pre-clipping gradient-norm maxima 54–3,904, and one
  spike of 7.6e7 (HPA seed 4); clipping absorbed them and HPA seed 4 has the best test PSNR. On this clip P3
  did not damage any run.
- **Early stopping**: VAE seeds 1 and 4 stopped at epochs 495 and 474; all HPA runs ran 500.
- **Image path** (`_im`): see P8. Spread over seeds is tiny (≤ 0.08 dB PSNR).
- **GP restoration** (`te_t`, HPA only: latent from the frame number by GP regression, then decoded):
  PSNR 27.97 ± 0.65 dB, SSIM 0.838 ± 0.005, i.e. 4.2 dB below encoding the same frames (`te_im` 32.20) and
  2.8 dB below the VAE encoding them (30.75). No naive baseline yet (e.g. copying or linearly interpolating
  the neighbouring training frames, which are 1–2 frames away with this random split).
- **Lengthscale**: 2.6–3.3 frames for seeds 2–5; seed 1 ended at 1.03 (3% above the lower bound 1, not
  flagged by the 1% rule) and has both the best `tr_t` (35.6 dB, others 30.8–31.5) and the worst `te_t`
  (26.8 dB, others 28.1–28.5): a short lengthscale fits the training latents more tightly and predicts
  held-out frames worse.
- Final training totals (VAE 0.85–0.86, HPA 0.11) are not comparable: different KL scales (B5).
- Masked ≈ unmasked here (B18: the mask removes 0.1% of pixels). Not comparable with pre-B1 numbers or
  with C3VD.

### T7. Full-size training on the CPU is not bit-reproducible, but evaluation is (tested)
Synthetic video, 16 frames of 336×336, all four models, 3 epochs, `OMP_NUM_THREADS=1`, CPU only
(`tests/equivalence/e2e_compare.py`): two runs of the *original* driver differ from epoch 0 in the 6th–7th
significant digit of the losses, and FID by about ±0.3. The refactored pipeline differs from the original by
the same amount. Ruled out: the EndoVAE encoder/decoder and LPIPS forward and backward at 336 px (identical
across repeats and across processes), the 112,896→16 linear layer (identical at any memory offset, with or
without `MKL_CBWR`). The cause is not found; the tiny 32×32 runs stay bit-exact (T4). Loading the same saved
models, the original and refactored programs give an identical 455-line run log and summary table (all
metrics, FID included). So the refactor's equivalence is shown by the tiny training runs plus this
evaluation check, not by comparing full training runs.

## Not yet measured
The effect of each fix on reconstruction metrics; whether the paper's tables came from the B1 code path;
a naive interpolation baseline for `te_t`; a β = 0 VAE control (P8).
