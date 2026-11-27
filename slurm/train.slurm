#!/bin/bash
#SBATCH --job-name=gpvae-legacy
#SBATCH --partition=ampere_gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
# Train and evaluate ONE (video, model, encoder, seed) per array task with `python -m gpvae_endo run
# configs/hpc.py`. The task's line in the job list says which. Submit from the repository root:
#
#   sbatch --account=$ACC --array=0-80%4 --export=$EXP,JOBS=slurm/jobs/c3vd.txt slurm/train.slurm
#
# (make the list with slurm/make_job_list.py; it prints the --array range.)
#
# Exported variables:
#   JOBS             job list (required): one line per run, "dataset video model encoder seed"; array index
#                    0 = first line (blank lines and # comments are skipped)
#   GPVAE_DATA_ROOT  where the NPZ files are: <root>/<dataset>/ (default $VSC_SCRATCH/gpvae-data)
#   RUNS_ROOT        where the run folders go (default <repo>/runs); one folder per line:
#                    <RUNS_ROOT>/<dataset>/<video>/<model>-<encoder>/seed<seed>/
#   KEEP_MODELS=1    keep the trained model (Saved_Models/*.pth, 340-450 MB) of a finished run
#   KEEP_FRAMES=1    keep the per-frame reconstruction PNGs (Recon/<video>/Frames/, 50-300 MB)
#                    (by default both are deleted once the run is evaluated, for the 50 GB scratch quota;
#                    the checkpoint is always deleted then; results, logs, plots and the video are kept)
#   GPVAE_EPOCHS     number of epochs (default 300, from configs/hpc.py); use a separate RUNS_ROOT for each value
#   GPVAE_MAX_RESUBMITS  how often a run may resubmit itself (default 10); GPVAE_RESUBMITS counts (internal)
# Arguments after the script name go to `python -m gpvae_endo run` (e.g. --checkpoint-interval 600).
#
# Checkpoints are always on. 30 minutes before the time limit (--deadline-margin 1800, room for the evaluation
# of a run that just finished training) the run saves and exits with code 85; this script then submits the
# same array index again and the new job continues where it stopped. A run that finished writes DONE in its
# folder; submitting its index again then does nothing.

REPO="${GPVAE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/slurm/common.sh"

EXIT_REQUEUE=85
: "${JOBS:?export JOBS=<job list>, e.g. slurm/jobs/c3vd.txt}"
JOBS=$(in_repo "$JOBS")
GPVAE_DATA_ROOT="${GPVAE_DATA_ROOT:-$VSC_SCRATCH/gpvae-data}"
RUNS_ROOT="${RUNS_ROOT:-$GPVAE_REPO/runs}"
GPVAE_RESUBMITS="${GPVAE_RESUBMITS:-0}"
GPVAE_MAX_RESUBMITS="${GPVAE_MAX_RESUBMITS:-10}"
KEEP_MODELS="${KEEP_MODELS:-0}"
KEEP_FRAMES="${KEEP_FRAMES:-0}"

INDEX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "$INDEX" ]]; then
    echo "ERROR: submit with --array=<line numbers, 0-based> (see slurm/make_job_list.py)" >&2
    exit 2
fi
if [[ ! -f "$JOBS" ]]; then
    echo "ERROR: job list $JOBS not found" >&2
    exit 2
fi
LINE=$(job_line "$JOBS" "$INDEX")
read -r DATASET VIDEO MODEL ENCODER SEED EXTRA <<< "$LINE" || true
if [[ -z "${SEED:-}" || -n "${EXTRA:-}" ]]; then
    echo "ERROR: line $((INDEX + 1)) of $JOBS is not 'dataset video model encoder seed': '$LINE'" >&2
    exit 2
fi

DATA_DIR="$GPVAE_DATA_ROOT/$DATASET"
RUN_DIR="$RUNS_ROOT/$DATASET/$VIDEO/$MODEL-$ENCODER/seed$SEED"

job_header train
echo "   run   $DATASET / $VIDEO / $MODEL + $ENCODER / seed $SEED${GPVAE_EPOCHS:+ / $GPVAE_EPOCHS epochs}"
echo "   data  $DATA_DIR"
echo "   out   $RUN_DIR"
echo "   resubmissions so far $GPVAE_RESUBMITS (max $GPVAE_MAX_RESUBMITS); extra arguments: $*"

if [[ -f "$RUN_DIR/DONE" ]]; then
    echo "== already finished ($RUN_DIR/DONE exists); delete the folder to train it again"
    exit 0
fi
NPZ="$DATA_DIR/$VIDEO 336 X Ym01sGM.npz"
if ! is_dry && [[ ! -f "$NPZ" ]]; then
    echo "ERROR: data file not found: $NPZ" >&2
    exit 2
fi
if ! is_dry; then
    nvidia-smi || echo "WARNING: nvidia-smi failed"
fi
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"   # the pipeline writes Logs/, Pics/, Recon/, Debug/, Saved_Models/, Experiment_Summaries.txt here
# The output folders (config_endo.py: FOLDER_LOGS, FOLDER_OUTPUT, ...). The pipeline creates them too; the
# original did not, and a fresh run folder then failed at the first log line.
mkdir -p Logs Pics Saved_Models Debug Recon

export GPVAE_VIDEO="$VIDEO" GPVAE_DATA_DIR="$DATA_DIR" GPVAE_MODELS="$MODEL" GPVAE_ENCODERS="$ENCODER" \
       GPVAE_SEED="$SEED"
ENTRY=(python -m gpvae_endo run "$GPVAE_REPO/configs/hpc.py" --checkpoint --deadline-margin 1800)

# The settings this job was started with, for a resubmission. Only exported variables reach a job on CalcUA,
# so everything is passed again explicitly. (Values must not contain commas.)
export_list() {
    local names=(VSC_SCRATCH VSC_DATA GPVAE_REPO GPVAE_ENV GPVAE_CACHE JOBS GPVAE_DATA_ROOT RUNS_ROOT
                 GPVAE_EPOCHS KEEP_MODELS KEEP_FRAMES GPVAE_MAX_RESUBMITS GPVAE_SBATCH GPVAE_DRY_RUN)
    local out="HOME,USER,TERM,PATH=/bin:/sbin" n
    for n in "${names[@]}"; do
        if [[ -n "${!n:-}" ]]; then
            [[ "${!n}" == *,* ]] && { echo "ERROR: $n contains a comma: ${!n}" >&2; return 1; }
            out+=",$n=${!n}"
        fi
    done
    echo "$out"
}

# Options that reproduce this job: same account, partition, time limit, name, and only this array index.
same_job_options() {
    local opts=()
    [[ -n "${SLURM_JOB_ACCOUNT:-}" ]] && opts+=(--account="$SLURM_JOB_ACCOUNT")
    [[ -n "${SLURM_JOB_PARTITION:-}" ]] && opts+=(--partition="$SLURM_JOB_PARTITION")
    [[ -n "${SLURM_JOB_NAME:-}" ]] && opts+=(--job-name="$SLURM_JOB_NAME")
    if [[ -n "${SLURM_JOB_START_TIME:-}" && -n "${SLURM_JOB_END_TIME:-}" ]]; then
        opts+=(--time=$(( (SLURM_JOB_END_TIME - SLURM_JOB_START_TIME + 59) / 60 )))  # minutes
    fi
    opts+=(--array="$INDEX")
    printf '%s\n' "${opts[@]}"
}

rc=0
run_python "${ENTRY[@]}" "$@" || rc=$?
echo "== run exited with code $rc at $(date '+%Y-%m-%d %H:%M:%S')"

if [[ "$rc" == "$EXIT_REQUEUE" ]]; then
    if (( GPVAE_RESUBMITS >= GPVAE_MAX_RESUBMITS )); then
        echo "ERROR: resubmission cap reached ($GPVAE_RESUBMITS of $GPVAE_MAX_RESUBMITS); not resubmitting." \
             "The checkpoint is kept: submit this index again by hand to continue." >&2
        exit "$EXIT_REQUEUE"
    fi
    mapfile -t SAME < <(same_job_options)
    EXPORTS="$(export_list),GPVAE_RESUBMITS=$(( GPVAE_RESUBMITS + 1 ))"
    is_dry || mkdir -p "$GPVAE_REPO/slurm/logs"
    out=$(cd "$GPVAE_REPO" && run_sbatch "${SAME[@]}" --dependency=afterany:"${SLURM_JOB_ID:-0}" \
          --export="$EXPORTS" "$GPVAE_REPO/slurm/train.slurm" "$@") \
        || { echo "ERROR: resubmission failed: $out" >&2; exit 1; }
    echo "$out"
    echo "== resubmitted ($(( GPVAE_RESUBMITS + 1 )) of max $GPVAE_MAX_RESUBMITS); this job ends with exit code 0"
    exit 0
fi

if [[ "$rc" != 0 ]]; then
    exit "$rc"
fi
date '+%Y-%m-%d %H:%M:%S' > DONE
echo "== finished: results in $RUN_DIR/Experiment_Summaries.txt"

# Free scratch space (50 GB quota). The results table, logs, plots (Pics/) and the reconstruction video
# (Debug/) are always kept; the evaluation is already done, so nothing else needs these files.
rm -f Saved_Models/*.ckpt.pt                                    # checkpoint: only for resuming, 1.1-1.4 GB
[[ "$KEEP_MODELS" == 1 ]] || rm -f Saved_Models/*.pth           # trained model: 340-450 MB
[[ "$KEEP_FRAMES" == 1 ]] || rm -rf Recon/*/Frames              # one PNG per frame: 50-300 MB
echo "== kept: $(du -sh . 2>/dev/null | cut -f1) (KEEP_MODELS=$KEEP_MODELS, KEEP_FRAMES=$KEEP_FRAMES)"
exit 0
