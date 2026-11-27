# Sourced by train.slurm and prepare.slurm (not run on its own).
#
# Jobs on CalcUA start with a minimal environment (only HOME, USER, TERM, PATH=/bin:/sbin, plus VSC_* and
# SLURM_*). This file rebuilds what the Python code needs. Settings come from variables exported with
#   sbatch --export=$EXP,NAME=value,...
#
#   GPVAE_REPO   this repository on the cluster (default: the directory sbatch was called from)
#   GPVAE_ENV    the container environment (default: $VSC_DATA/gpvae_env, the one setup.sh exports)
#   GPVAE_CACHE  downloaded weights and caches (default: $VSC_SCRATCH/gpvae-cache, shared with the rewrite copy)
#   GPVAE_DRY_RUN=1  print the commands instead of running them (for testing the scripts)
#   GPVAE_DRY_EXIT   in a dry run, the exit code the Python command pretends to return (default 0)

set -euo pipefail

# If sbatch --export named only our variables, PATH may be missing; the system tools must still be found.
export PATH="${PATH:+$PATH:}/usr/bin:/bin:/usr/sbin:/sbin"

if [[ -z "${VSC_SCRATCH:-}" ]]; then
    echo "ERROR: VSC_SCRATCH is not set: submit with --export=\$EXP,... (EXP from setup.sh)" >&2
    exit 2
fi
GPVAE_REPO="${GPVAE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
GPVAE_ENV="${GPVAE_ENV:-${VSC_DATA:-$VSC_SCRATCH}/gpvae_env}"
GPVAE_CACHE="${GPVAE_CACHE:-$VSC_SCRATCH/gpvae-cache}"
GPVAE_DRY_RUN="${GPVAE_DRY_RUN:-0}"

export PATH="$GPVAE_ENV/bin:$PATH"
export PYTHONPATH="$GPVAE_REPO${PYTHONPATH:+:$PYTHONPATH}"  # runs start in their own folder, not the repo
export TORCH_HOME="$GPVAE_CACHE/torch"         # LPIPS VGG16 and FID Inception weights
export HF_HOME="$GPVAE_CACHE/huggingface"      # only used if weights/dinov2_base.pth is missing
export MPLCONFIGDIR="$GPVAE_CACHE/matplotlib"  # matplotlib's cache; the home directory has a small quota
export MPLBACKEND=Agg                          # no display on compute nodes
export PYTHONUNBUFFERED=1                      # log lines appear in the job log as they are printed
export PYTHONIOENCODING=utf-8                  # the logs contain non-ASCII characters (e.g. a check mark); jobs have no locale
# CPU threads (PyTorch, OpenCV, NumPy): the CPUs Slurm gave this job, not all 64 of the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"

is_dry() { [[ "$GPVAE_DRY_RUN" == 1 ]]; }

# Line <index> (0-based) of a list file, not counting blank lines and lines starting with #. Windows line
# endings (a list edited or packed on Windows) are removed.
job_line() { tr -d '\r' < "$1" | grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' | sed -n "$(( $2 + 1 ))p" || true; }

# A path relative to the repository made absolute (/x and, for tests under Git Bash, C:/x are kept).
in_repo() { if [[ "$1" = /* || "$1" =~ ^[A-Za-z]:/ ]]; then echo "$1"; else echo "$GPVAE_REPO/$1"; fi; }

# Print a command on one line; arguments with white space (or empty ones) in single quotes.
show() {
    local a out=()
    for a in "$@"; do
        if [[ -z "$a" || "$a" =~ [[:space:]] ]]; then out+=("'$a'"); else out+=("$a"); fi
    done
    echo "${out[*]}"
}

# Run the Python step. In a dry run, print it and return GPVAE_DRY_EXIT.
run_python() {
    if is_dry; then
        printf 'DRY-RUN python: '; show "$@"
        return "${GPVAE_DRY_EXIT:-0}"
    fi
    "$@"
}

# sbatch, found on PATH or given as GPVAE_SBATCH. In a dry run, print the call and pretend job 999 was made.
run_sbatch() {
    if is_dry; then
        printf 'DRY-RUN sbatch: '; show sbatch "$@"
        echo "Submitted batch job 999"
        return 0
    fi
    local bin="${GPVAE_SBATCH:-$(command -v sbatch || true)}"
    if [[ -z "$bin" ]]; then
        echo "ERROR: sbatch not found on PATH ($PATH); export GPVAE_SBATCH=/path/to/sbatch" >&2
        return 1
    fi
    "$bin" "$@"
}

job_header() {
    echo "== $1: job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ (array task $SLURM_ARRAY_TASK_ID)}" \
         "on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "   repo  $GPVAE_REPO"
    echo "   env   $GPVAE_ENV"
    echo "   cache $GPVAE_CACHE"
    if ! is_dry; then
        if [[ ! -x "$GPVAE_ENV/bin/python" ]]; then
            echo "ERROR: no Python at $GPVAE_ENV/bin/python: export GPVAE_ENV (setup.sh does)" >&2
            exit 1
        fi
        echo "   python $(command -v python): $(python --version 2>&1)"
    fi
    mkdir -p "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR"
}
