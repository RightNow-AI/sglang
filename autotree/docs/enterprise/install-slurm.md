# SLURM install

The templates in `deploy/slurm/` package the current CPU TreeKV server for a
module-based lab cluster.

Prerequisites are a shared checkout, Python 3.12 and Rust 1.93 modules, outbound
access for Python packages/model weights (or pre-populated caches), and an open
port between clients and allocated compute nodes.

```bash
cd /shared/projects/autotree
AUTOTREE_PYTHON_MODULE=python/3.12 \
AUTOTREE_RUST_MODULE=rust/1.93 \
AUTOTREE_MODEL=gpt2 \
sbatch deploy/slurm/single-node.sbatch
```

After the log reports Uvicorn listening, find the node with `squeue` and point
the client at `http://NODE:8000/v1`.

For independent replicas on multiple nodes:

```bash
sbatch --nodes=4 deploy/slurm/multi-node.sbatch
```

The multi-node job writes its node list in the submission directory. Put those
node endpoints behind the lab's load balancer or distribute requests in the
client. It does not provide model sharding, GPU-direct KV transfer, or a shared
distributed scheduler; those blueprint items remain roadmap work.

See `deploy/slurm/README.md` for environment overrides and the current GPU
limitation.
