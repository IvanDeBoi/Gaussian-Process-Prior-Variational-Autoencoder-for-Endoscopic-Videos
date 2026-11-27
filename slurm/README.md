# Training on the CalcUA cluster

This trains **every combination** on **every video**, one GPU job per run:

- **Models × encoders (9):** `VAE`, `GPVAE_HPA` and `GPVAE_SPA`, each with `EndoVAEEnc`, `GastroEnc` and
  `GaLoRaEnc` (decoder `EndoVAEDec`).
- **Videos:** 9 per dataset. Each video gets its own models, as in the paper; nothing is trained on several
  videos at once.
- **Settings:** 300 epochs, 20% of each video's frames held out for testing, seed 42. Everything else is
  `config_endo.py`'s `is_on_HPC` block: batch 64, latent 256, LPIPS at 256 px, warm-up 50, KL ramp 50.
  The settings file is `configs/hpc.py`.

That is 9 runs per video and 81 per dataset. The datasets are done **one at a time**, because of the 50 GB
scratch quota:

1. **Kvasir-Capsule** (steps 4–8): the 9 videos of `metadata_selected.json`, list ready in
   `slurm/videos_kvasir.txt`.
2. **C3VD** (step 9).
3. **HyperKvasir** (step 9).

Steps 1–3 are done once.

The code is the original (`aaf8b6d`), restructured, with the B1 and B2 fixes (see the main README, "How this
version differs from the original code").

**What has been tested** (on the laptop, not on the cluster):
- the scripts in dry-run mode (`tests/test_hpc.py`);
- `train.slurm` for real, with a fake Slurm environment and 2 epochs: VAE + `EndoVAEEnc`, HPA + `EndoVAEEnc`
  and VAE + `GastroEnc` trained, were evaluated, wrote `DONE`, cleaned up, and `collect_results.py` read them;
- `prepare.slurm` for real, on 20 frames of a Kvasir video, from a list written like `videos_kvasir.txt`.

**Nothing in this folder has run on the cluster yet**, so run the smoke test (step 5) first.

The cluster setup it relies on is yours from the `rewrite` runs:
- the environment (container) at `$VSC_DATA/gpvae_env`;
- `setup.sh` (which sets `GPVAE_ENV`, `EXP` and `ACC`);
- the weight cache in `$VSC_SCRATCH/gpvae-cache`;
- the SSH key `id_rsa_vsc`.

This copy lives next to the `rewrite` one, in `$VSC_SCRATCH/gpvae-legacy`, so the two don't mix.

---

## 1. Once, on the laptop: pack and upload the code and weights

In PowerShell, from the repository folder, on branch `main`. `git archive` packs only
committed files, so commit first:

```bash
cd Gaussian-Process-Prior-Variational-Autoencoder-for-Endoscopic-Videos
```

```bash
git -c core.autocrlf=false archive --format=tar.gz -o gpvae-legacy-code.tar.gz main
```

```bash
tar -czf gpvae-legacy-weights.tar.gz -C Models/Weights dinov2_base.pth Duck16mod_Spec_99_new.pt
```

- `dinov2_base.pth` is needed by `GastroEnc` and `GaLoRaEnc`. Compute nodes may have no internet, so the
  code's download fallback can't be relied on.
- `Duck16mod_Spec_99_new.pt` is the reflection detector, needed to make the data files (step 4).
- `core.autocrlf=false` keeps the job scripts' Linux line endings.

Upload. The files go straight to scratch: the home folder holds only about 3 GB.

```bash
scp -i $env:USERPROFILE\.ssh\id_rsa_vsc gpvae-legacy-code.tar.gz gpvae-legacy-weights.tar.gz vsc21309@login.hpc.uantwerpen.be:/scratch/antwerpen/213/vsc21309/
```

## 2. Once, on the cluster: unpack and set up

Log in (`ssh -i $env:USERPROFILE\.ssh\id_rsa_vsc vsc21309@login.hpc.uantwerpen.be`), then:

```bash
mkdir -p $VSC_SCRATCH/gpvae-legacy/weights && cd $VSC_SCRATCH/gpvae-legacy && tar -xzf $VSC_SCRATCH/gpvae-legacy-code.tar.gz && tar -xzf $VSC_SCRATCH/gpvae-legacy-weights.tar.gz -C weights && mkdir -p slurm/logs && rm $VSC_SCRATCH/gpvae-legacy-code.tar.gz $VSC_SCRATCH/gpvae-legacy-weights.tar.gz
```

`slurm/logs` must exist: Slurm does not create it, and a job without it fails at once.

**A `setup.sh` for this copy.** It is the same as your existing one, except that it changes into
`gpvae-legacy`:

```bash
sed 's#gpvae-method#gpvae-legacy#' $VSC_SCRATCH/gpvae-method/setup.sh > $VSC_SCRATCH/gpvae-legacy/setup.sh && cat $VSC_SCRATCH/gpvae-legacy/setup.sh
```

After each login:

```bash
source $VSC_SCRATCH/gpvae-legacy/setup.sh
```

All later commands are typed in `$VSC_SCRATCH/gpvae-legacy` (where `setup.sh` takes you).

**Check the environment once** (on the login node; it uses no GPU). This imports everything the pipeline
uses. It also downloads the LPIPS (VGG16) and FID (Inception) weights into the shared cache if they aren't
there yet: compute nodes may have no internet, and this branch always computes FID.

```bash
TORCH_HOME=$VSC_SCRATCH/gpvae-cache/torch $GPVAE_ENV/bin/python -c "import gpvae_endo.pipeline, scripts.prepare_dataset, lpips; lpips.LPIPS(net='vgg'); from torchmetrics.image.fid import FrechetInceptionDistance as F; F(feature=2048); print('environment OK')"
```

If it reports a missing module, note which one: nothing can be `pip install`ed into the container (its
README), so it has to be rebuilt.

**Updating the code later** (after a new commit): make a new `gpvae-legacy-code.tar.gz` (step 1), upload it,
and unpack it over the old copy with `cd $VSC_SCRATCH/gpvae-legacy && tar -xzf $VSC_SCRATCH/gpvae-legacy-code.tar.gz`.
That replaces the code files and leaves `runs/`, `weights/`, `setup.sh` and the logs alone. Don't do it while
jobs are running.

## 3. Where things go

| What | Where |
|---|---|
| Code | `$VSC_SCRATCH/gpvae-legacy` |
| Data files | `$VSC_SCRATCH/gpvae-data/<dataset>/<video> 336 X Ym01sGM.npz` (dataset = `kvasir`, `c3vd`, `hyperkvasir`) |
| Raw videos (only while making the data files) | `$VSC_SCRATCH/videos/<dataset>/` |
| One run | `runs/<dataset>/<video>/<model>-<encoder>/seed42/`: the files the original wrote into the folder it ran from: `Experiment_Summaries.txt` (the scores), `Logs/` (everything printed), `Pics/` (plots), `Debug/<video>/` (example frames and the original-vs-reconstructed video), and `DONE` when finished. `Saved_Models/` and `Recon/<video>/` are emptied when the run finishes, unless `KEEP_MODELS=1` / `KEEP_FRAMES=1` |
| Job logs | `slurm/logs/gpvae-legacy_<array job>_<index>.out` (training), `slurm/logs/gpvae-prepare_<array job>_<index>.out` (data) |
| Mask previews | `Resulting_Masks/<video>_336/` (every 10th frame) |

### Disk space (50 GB on scratch)

The container solves the file-count limit, but scratch is still 50 GB. If nothing were deleted, one dataset
(81 runs) would need about 160 GB:

| Item | Size | What happens to it |
|---|---|---|
| Checkpoint `Saved_Models/*.ckpt.pt` | 1.1–1.4 GB per run | needed only while training; deleted when the run finishes |
| Model `Saved_Models/*.pth` | 340 MB (`EndoVAEEnc`) to 450 MB (ViT) per run | deleted when the run finishes, unless `KEEP_MODELS=1` |
| Per-frame PNGs `Recon/<video>/Frames/` | 110–165 KB per frame: 50–120 MB per Kvasir run | deleted when the run finishes, unless `KEEP_FRAMES=1` |
| Scores, logs, plots, comparison video | about 10 MB (VAE); GP models more, because of one large GP-posterior plot (75 MB on an 80-frame test video) | kept |
| Data file of one video | about 1.45 MB per frame | kept until you delete it |

- **Data files:** Kvasir-Capsule about 6 GB (4,033 frames); C3VD about 5 GB; HyperKvasir depends on the
  frames chosen (2,000 frames per video would make about 26 GB).
- **Running jobs:** about 2 GB each while training (checkpoint + model + frames), so about 8 GB with 4 at
  a time.
- **Finished runs:** a few GB per dataset. Check the real size after the smoke test (step 5).

So:

1. **One dataset at a time.** Train it, collect and download the results (step 8), then delete its data
   files and runs before preparing the next one.
2. **Check the space** before each dataset: `du -sh $VSC_SCRATCH/* | sort -h` (and `myquota`).
3. **Keep models only where you need them** (`KEEP_MODELS=1`): all 81 models of one dataset would take about
   33 GB.

---

## 4. Kvasir-Capsule: make the data files

**The list is ready.** `slurm/videos_kvasir.txt` has the 9 chosen videos (`metadata_selected.json`). Each uses
its longest "Normal clean mucosa" segment, whole: 270–708 frames, 4,033 frames together. That mirrors C3VD,
where each video is one short, continuous, clean recording of 208–721 frames. Line format:

```
dataset  name  source  [first  last]
```

`first`/`last` are the 0-based frame range, both inclusive; `$VSC_SCRATCH` may be written literally.

**Upload the videos** (PowerShell, about 4.8 GB). The 9 videos are in `Data\kvasir` on the laptop. As long as
`videos/kvasir` does not exist on the cluster, scp creates it with the files inside (if it does exist, they
end up one level deeper, in `videos/kvasir/kvasir`):

```bash
scp -r -i $env:USERPROFILE\.ssh\id_rsa_vsc C:\MyStuff\KDG_ACS_PROJECTS\gpvae-method\Data\kvasir vsc21309@login.hpc.uantwerpen.be:/scratch/antwerpen/213/vsc21309/videos/kvasir
```

On the cluster, check that the 9 `.mp4` files are there: `ls -la $VSC_SCRATCH/videos/kvasir`.

**Make the data files:** one CPU job per video (array index = video line, comments skipped).

```bash
sbatch --account=$ACC --array=0-8 --export=$EXP,VIDEOS=slurm/videos_kvasir.txt slurm/prepare.slurm
```

What `scripts/make_npz.py` does with each video:
- reads the frames `first..last`;
- keeps them at 336 × 336 (Kvasir-Capsule frames already are; other sizes get the C3VD treatment, shorter
  side to 336 and a centre crop);
- computes the reflection mask with the original `prepare_dataset.py` code;
- writes `$VSC_SCRATCH/gpvae-data/kvasir/kvasir_<id> 336 X Ym01sGM.npz`.

DUCKNet takes about 0.4 s per frame on a CPU node, so each video takes a few minutes. When `squeue -u $USER`
is empty:

```bash
tail -n 3 slurm/logs/gpvae-prepare_*.out
```

```bash
ls -la $VSC_SCRATCH/gpvae-data/kvasir
```

Each log should end with `N frames (numbers A..B), valid pixels ..., total ... s`, where A..B is the line's
range and the valid share is close to 1 (the share of pixels not marked as reflections; 1.0000 on the 20
test frames). There should be 9 data files. Look at a few mask previews too (`Resulting_Masks/`). Then delete the videos from the cluster
(they stay on the laptop):

```bash
rm -r $VSC_SCRATCH/videos/kvasir
```

## 5. Smoke test first (3 epochs, all 9 combinations, one video)

This checks the environment, the data, the GPU memory of every model and encoder, and the whole evaluation.
It writes to a separate folder, `runs_smoke`, so it can't get in the way of the real runs. First make the
job list (it prints the number of runs and the `sbatch` line for the real runs):

```bash
$GPVAE_ENV/bin/python slurm/make_job_list.py kvasir
```

The list is ordered by seed, model, encoder, then video, so lines 0, 9, 18, …, 72 are the 9 combinations on
the first video (`kvasir_0728084c8da942d9`, 435 frames):

```bash
sbatch --account=$ACC --array=0,9,18,27,36,45,54,63,72%4 --time=01:00:00 --export=$EXP,JOBS=slurm/jobs/kvasir.txt,GPVAE_EPOCHS=3,RUNS_ROOT=$VSC_SCRATCH/gpvae-legacy/runs_smoke slurm/train.slurm
```

When they are done (`squeue -u $USER` shows nothing):

```bash
sacct -u $USER -S today --format=JobID%20,JobName%15,State,ExitCode,Elapsed,MaxRSS
```

```bash
$GPVAE_ENV/bin/python slurm/collect_results.py runs_smoke
```

```bash
du -sh runs_smoke/kvasir/*/*/seed42
```

- All 9 should be `COMPLETED 0:0`, and the table should have 9 rows.
- `grep -l "out of memory" slurm/logs/gpvae-legacy_*.out` should find nothing.
- `grep "t/epoch" -A 3 slurm/logs/gpvae-legacy_*.out` gives the seconds per epoch; this video has 348 training
  frames. Use the slowest value for the time limit in step 6.
- `du` shows what a finished run keeps; multiply by 81 for the dataset.

Then remove the test: `rm -r runs_smoke`.

## 6. The real runs

```bash
sbatch --account=$ACC --array=0-80%4 --export=$EXP,JOBS=slurm/jobs/kvasir.txt slurm/train.slurm
```

- `%4` lets at most 4 of these jobs run at once: the A100 node has 4 GPUs.
- The list is ordered by model, then encoder, then video: lines 0–26 are the `VAE` runs, 27–53 `GPVAE_HPA`,
  54–80 `GPVAE_SPA`; within each model the 9 `EndoVAEEnc` runs come first, then `GastroEnc`, then `GaLoRaEnc`.
  So the first finished results are the VAE baselines.

**Time.** The job's time limit is 12 hours (`#SBATCH --time` in `train.slurm`). A shorter limit lets Slurm
start jobs sooner: add for example `--time=04:00:00` to the `sbatch` line if 300 × the slowest `t/epoch` of the
smoke test (for the longest video, 708 frames, scale it by 566 / 348) plus 45 minutes fits. Checkpointing is
always on, so a limit that turns out too short costs only a little time:
- 30 minutes before the limit, a run that is still training saves and exits with code 85;
- the job then submits the same index again, and the new job continues where it stopped (up to 10 times).

## 7. While it runs

| To see | Command |
|---|---|
| Your jobs | `squeue -u $USER` |
| One log, live | `tail -f slurm/logs/gpvae-legacy_<job>_<index>.out` |
| What finished or failed | `sacct -u $USER -S today --format=JobID%20,State,ExitCode,Elapsed` |
| How many runs are finished | `ls runs/kvasir/*/*/seed42/DONE \| wc -l` |
| Disk space | `myquota`, `du -sh $VSC_SCRATCH/* \| sort -h` |

**A job that failed:**
1. Read its log.
2. Fix the cause.
3. Submit just that index again: `sbatch --account=$ACC --array=<index> --export=$EXP,JOBS=slurm/jobs/kvasir.txt slurm/train.slurm`.

A run whose training had finished (for example, one killed by the time limit during the evaluation) still has
its checkpoint, so it resumes from it and only does the evaluation again. Submitting the whole list again is
also safe, because every finished run (with `DONE`) is skipped.

**Disk full** (`No space left on device` or `Disk quota exceeded` in a log): free space, then submit the
failed indexes again.

**To train a run again from scratch:** delete its folder (`rm -r runs/kvasir/<video>/<model>-<encoder>/seed42`).

## 8. Results, download, clean up

```bash
$GPVAE_ENV/bin/python slurm/collect_results.py
```

This writes `runs/results.csv`, with one row per run: dataset, video, model, encoder, seed, and all the
summary columns (RMSE, FID, SSIM and PSNR, for training and test frames, image- and time-encoded). It also
prints a mean ± standard deviation over the videos of each dataset, for every model and encoder. Runs still
training or failed are listed at the end; when that list is empty, the dataset is complete.

Pack the results on the cluster (into scratch, not the small home folder) and download them (PowerShell):

```bash
ssh -i $env:USERPROFILE\.ssh\id_rsa_vsc vsc21309@login.hpc.uantwerpen.be "cd /scratch/antwerpen/213/vsc21309/gpvae-legacy && tar -czf results-kvasir.tar.gz runs/kvasir runs/results.csv slurm/logs slurm/jobs Resulting_Masks"
```

```bash
scp -i $env:USERPROFILE\.ssh\id_rsa_vsc vsc21309@login.hpc.uantwerpen.be:/scratch/antwerpen/213/vsc21309/gpvae-legacy/results-kvasir.tar.gz .
```

Unpack it on the laptop (`tar -xzf results-kvasir.tar.gz`) and check it opens. Only then free the cluster for
the next dataset:

```bash
cd $VSC_SCRATCH/gpvae-legacy && rm -r results-kvasir.tar.gz runs/kvasir runs/results.csv Resulting_Masks slurm/logs/* $VSC_SCRATCH/gpvae-data/kvasir
```

## 9. The next datasets: C3VD, then HyperKvasir

The same steps 4–8, with `kvasir` replaced by `c3vd` or `hyperkvasir`. What differs:

- **C3VD:** the data files already exist on the laptop (`Data/`, one per settings file in `configs/`). Pack and
  upload them instead of videos (PowerShell, from the repository folder):

  ```bash
  $v = (Get-ChildItem configs\config_c*.py).BaseName -replace '^config_',''; tar -cf c3vd-npz.tar -C Data ($v | ForEach-Object { "$_ 336 X Ym01sGM.npz" })
  ```

  ```bash
  scp -i $env:USERPROFILE\.ssh\id_rsa_vsc c3vd-npz.tar vsc21309@login.hpc.uantwerpen.be:/scratch/antwerpen/213/vsc21309/
  ```

  On the cluster: `mkdir -p $VSC_SCRATCH/gpvae-data/c3vd && tar -xf $VSC_SCRATCH/c3vd-npz.tar -C $VSC_SCRATCH/gpvae-data/c3vd && rm $VSC_SCRATCH/c3vd-npz.tar`.
  For other C3VD videos, write a list like `videos_kvasir.txt` whose source is a folder of frame images
  (`rgb/`), and use `prepare.slurm` as in step 4.
- **HyperKvasir:** write `slurm/videos_hyperkvasir.txt` like `videos_kvasir.txt` (dataset `hyperkvasir`).
  Without `first`/`last` a video is used from frame 0, at most 2,000 frames (`--max-frames` after the script
  name changes that). Wider frames are scaled to 336 px high and centre-cropped, which cuts the left and right
  edges.

The job list (`make_job_list.py c3vd`), the smoke test (lines 0, 9, …, 72 of the new list) and the real runs
work the same way.

## 10. Reading the results: what they can and can't show

- **This is the original method**, with B1 (scores on the whole frame) and B2 (best epoch restored) fixed.
- **Not a like-for-like comparison:** B4 (HPA's KL never reaches the encoder) and B5 (each model weighs the KL
  differently) are kept on purpose (see `docs/findings.md`), so VAE vs GP-VAE differences can't be read as
  "the time model helps".
- **Training can blow up** (P3, seen with `EndoVAEEnc`). Look at the loss lines in the log of any run with
  odd scores.
- **FID is unreliable** with few frames: a Kvasir video has only about 54–142 test frames (B8, B20).
- **Time-encoded scores (`_t`)** exist only for the GP models; they show how well a frame is predicted from its
  frame number alone.
- **One seed (42).** For spreads over seeds: `$GPVAE_ENV/bin/python slurm/make_job_list.py kvasir --seeds 42 1 2 3 4`
  (5 × 81 runs; the finished seed-42 runs have `DONE` and are skipped).
- **Kvasir video `kvasir_d7a271f233ba4a40`:** its longest "Normal clean mucosa" segment (frames 10808–11175) is
  full of bubbles and debris. It was kept on purpose; mention it when its scores differ from the others.
