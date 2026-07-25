# Model sources

## Hugging Face and local directories

`autotree serve --engine treekv --model VALUE` passes `VALUE` to Transformers'
`AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM.from_pretrained`. It can
therefore use a Hugging Face model identifier such as `gpt2` or a local
Transformers-compatible directory.

For private Hugging Face repositories, provide `HF_TOKEN` through a Kubernetes
Secret using the chart's `extraEnv` or `extraEnvFrom` values. Mount a persistent
cache at `/home/autotree/.cache/huggingface` if model downloads must survive pod
replacement; the chart's default cache is an `emptyDir`.

Air-gapped operation requires model files and Python/container dependencies to
be mirrored before installation. Point `--model` at the mounted local model
directory.

## S3

Direct `s3://` model identifiers, S3 credentials, and an S3 download layer are
not implemented in the current server. Do not pass an S3 URI to `--model`.

Today, use an init container or an external artifact sync process to copy an S3
object prefix into a mounted Transformers-compatible local directory, then set
the chart's `model` value to that directory. Native S3 resolution, checksums,
and atomic version promotion remain roadmap work.
