# SLURM templates

The templates launch the current `autotree serve --engine treekv` CPU reference
server. They load configurable Python and Rust modules, create a reusable venv,
build the scheduler wheel, install the local core/server packages, and start the
HTTP process with `srun`.

```bash
AUTOTREE_PYTHON_MODULE=python/3.12 \
AUTOTREE_RUST_MODULE=rust/1.93 \
AUTOTREE_MODEL=gpt2 \
sbatch deploy/slurm/single-node.sbatch
```

`multi-node.sbatch` requires the checkout and venv to live on a filesystem
shared by every allocated node. It launches one independent server on port 8000
per node and writes the allocated host list to
`autotree-$SLURM_JOB_ID-nodes.txt`. It does not shard a model or move KV state
between nodes; the repository does not yet contain that distributed runtime.

The templates intentionally do not request GPUs because the current packaged
TreeKV server uses the CPU model executor. Add site-specific `#SBATCH --gres`
only with a separately validated GPU serving image/runtime.
