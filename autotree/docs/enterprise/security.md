# Security and governance

Enterprise middleware is disabled when no related environment variables are
set, so the existing quickstart remains unauthenticated. When enabled, it
protects `/v1/*`. `/health`, `/metrics`, and `/playground` remain unauthenticated
operational surfaces and should be limited with NetworkPolicy, ingress policy,
or a private Service.

## API keys

Set `AUTOTREE_API_KEYS` to a JSON object mapping secrets to tenant names:

```bash
export AUTOTREE_API_KEYS='{"secret-value":"research-team"}'
```

For Kubernetes, avoid putting secrets in Helm values. Create a JSON file and a
Secret, then point the chart at it:

```bash
printf '%s' '{"secret-value":"research-team"}' >api-keys.json
kubectl -n autotree create secret generic autotree-api-keys \
  --from-file=api-keys.json
helm upgrade autotree deploy/helm/autotree -n autotree \
  --reuse-values \
  --set enterprise.apiKeys.existingSecret=autotree-api-keys
```

Clients send `X-API-Key: secret-value`. The file can also be selected directly
with `AUTOTREE_API_KEYS_FILE`; an environment JSON mapping overrides duplicate
keys from the file.

## OIDC bearer tokens

OIDC is enabled only when all three values are present:

```text
AUTOTREE_OIDC_JWKS_URL=https://issuer.example/.well-known/jwks.json
AUTOTREE_OIDC_ISSUER=https://issuer.example/
AUTOTREE_OIDC_AUDIENCE=autotree
AUTOTREE_OIDC_ALGORITHMS=RS256
```

The middleware obtains the signing key from the JWKS URL and verifies the token
signature, `iss`, `aud`, expiration, and required `sub`. Tenant identity is the
first non-empty claim among `tenant`, `tenant_id`, `tid`, and `sub`. SAML is not
implemented; use an identity provider that bridges SAML to OIDC or treat native
SAML support as roadmap.

## Per-tenant generated-token quotas

Configure fixed-window limits with a JSON tenant-to-token mapping:

```bash
export AUTOTREE_TENANT_QUOTAS='{"research-team":50000}'
export AUTOTREE_QUOTA_WINDOW_SECONDS=60
```

The middleware reserves the maximum generated-token amount before admission
(`tree.budget_tokens`, otherwise the resolved token limit) and settles the
reservation to actual generated tokens after the response or stream. An
exhausted quota returns HTTP 429 with error code
`tenant_token_quota_exceeded` and `Retry-After`.

Quota state is in process memory. With multiple replicas, each pod enforces its
own window; this is not yet a cluster-wide or durable quota ledger. A shared
atomic quota store is roadmap work.

## Audit log

Set `AUTOTREE_AUDIT_LOG=/var/log/autotree/audit.jsonl`. Each completed `/v1/*`
request appends one compact JSON line:

```json
{"timestamp":"2026-07-19T20:00:00Z","tenant":"research-team","route":"/v1/tree/completions","model":"gpt2","tokens":32,"status":200}
```

`tokens` is the actual generated-token count observed from engine events. The
file is opened with append mode for every record and writes are serialized
inside the process. Helm can mount an empty directory or PVC. This is an
append-only application log, not tamper-evident storage: ship it to immutable
external storage for retention and integrity controls.

The chart's generated PVC defaults to `ReadWriteOnce`. For more than one
replica, use a storage class that supports the required multi-pod access mode or
ship each pod's audit file independently; do not assume a single RWO claim can
be mounted across nodes.

## Implemented versus roadmap

| Control | Status |
| --- | --- |
| API-key authentication | Implemented for `/v1/*` |
| OIDC JWT verification through JWKS, issuer, and audience | Implemented |
| Per-process tenant generated-token quotas | Implemented |
| Append-only JSONL request audit | Implemented |
| Kubernetes RBAC objects for AutoTree application roles | Roadmap; the server has no role model today |
| Native SAML | Roadmap |
| Cluster-wide durable quota accounting | Roadmap |
| Tamper-evident audit archive and retention policy automation | Roadmap |
| SOC 2 certification | Roadmap; no certification claim is made |

SOC 2 preparation still requires organization-level controls outside this
repository: access reviews, change management, incident response, vendor risk,
backup/restore evidence, retention policies, monitoring ownership, and an audit
period with an independent assessor.
