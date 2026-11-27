#!/bin/bash
#SBATCH --job-name=gpvae-prepare
#SBATCH --partition=zen2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
# Make the NPZ data file of ONE video per array task with scripts/make_npz.py (CPU job; DUCKNet took
# 0.38 s per frame on zen2, so 2,000 frames take about 13 minutes). Submit from the repository root:
#
#   sbatch --account=$ACC --array=0-8 --export=$EXP,VIDEOS=slurm/videos_kvasir.txt slurm/prepare.slurm
#
# Exported variables:
#   VIDEOS           list file (required), one line per video (array index 0 = first line; blank lines and # comments skipped):
#                        dataset  name  source  [first  last]
#                    source = a video file or a folder of numbered frame images; first/last = 0-based,
#                    inclusive frame range (video only; default: from frame 0, at most 2,000 frames).
#                    No spaces inside a field; "$VSC_SCRATCH" may be written literally in the source.
#   GPVAE_DATA_ROOT  output: <root>/<dataset>/<name> 336 X Ym01sGM.npz (default $VSC_SCRATCH/gpvae-data)
# Arguments after the script name go to make_npz.py (e.g. --max-frames 3000, --force).

REPO="${GPVAE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/slurm/common.sh"

: "${VIDEOS:?export VIDEOS=<list file>}"
VIDEOS=$(in_repo "$VIDEOS")
if [[ ! -f "$VIDEOS" ]]; then
    echo "ERROR: video list $VIDEOS not found" >&2
    exit 2
fi
GPVAE_DATA_ROOT="${GPVAE_DATA_ROOT:-$VSC_SCRATCH/gpvae-data}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(job_line "$VIDEOS" "$INDEX")
read -r DATASET NAME SOURCE FIRST LAST EXTRA <<< "$LINE" || true
if [[ -z "${SOURCE:-}" || -n "${EXTRA:-}" ]]; then
    echo "ERROR: line $((INDEX + 1)) of $VIDEOS is not 'dataset name source [first last]': '$LINE'" >&2
    exit 2
fi

SOURCE="${SOURCE//\$VSC_SCRATCH/$VSC_SCRATCH}"   # the list may write $VSC_SCRATCH literally
job_header prepare
echo "   $DATASET / $NAME from $SOURCE ${FIRST:+frames $FIRST..${LAST:-end}}"

ARGS=(python "$GPVAE_REPO/scripts/make_npz.py" --name "$NAME" --out "$GPVAE_DATA_ROOT/$DATASET")
if [[ -d "$SOURCE" ]]; then
    ARGS+=(--frames "$SOURCE")
else
    ARGS+=(--video "$SOURCE")
    [[ -n "${FIRST:-}" ]] && ARGS+=(--first "$FIRST")
    [[ -n "${LAST:-}" ]] && ARGS+=(--last "$LAST")
fi
cd "$GPVAE_REPO"   # mask previews go to Resulting_Masks/ here
run_python "${ARGS[@]}" "$@"
