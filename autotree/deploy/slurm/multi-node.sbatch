#!/usr/bin/env bash
#SBATCH --job-name=autotree-serve-nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=autotree-multinode-%j.out

set -euo pipefail

module purge
module load "${AUTOTREE_PYTHON_MODULE:-python/3.12}"
module load "${AUTOTREE_RUST_MODULE:-rust/1.93}"

repo_root="${AUTOTREE_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
venv="${AUTOTREE_VENV:-$repo_root/.venv-slurm}"
model="${AUTOTREE_MODEL:-gpt2}"

# Bootstrap once on the submission node. This requires a shared filesystem.
if [[ ! -x "$venv/bin/autotree" ]]; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install --upgrade pip maturin
  "$venv/bin/maturin" build \
    --release \
    --features python \
    --manifest-path "$repo_root/scheduler/Cargo.toml" \
    --out "$repo_root/dist"
  scheduler_wheel="$(find "$repo_root/dist" -maxdepth 1 -name 'autotree_scheduler-*.whl' -print -quit)"
  "$venv/bin/pip" install "$repo_root/core[engine]" "$repo_root/serve" "$scheduler_wheel"
fi

scontrol show hostnames "$SLURM_JOB_NODELIST" > "autotree-${SLURM_JOB_ID}-nodes.txt"

# Current autotree-serve has no distributed runtime. This starts one independent
# server per node; clients or a cluster load balancer distribute requests.
exec srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  "$venv/bin/autotree" serve \
  --engine treekv \
  --model "$model" \
  --host 0.0.0.0 \
  --port "${AUTOTREE_PORT:-8000}" \
  --kv-branch-headroom "${AUTOTREE_KV_BRANCH_HEADROOM:-1.5}"
