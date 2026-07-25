# Security policy

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository
(Security tab, "Report a vulnerability"). Do not open a public issue for
security problems.

Include reproduction steps and the affected component. You will get an
acknowledgement within a week.

## Scope notes

- `autotree serve` binds to 127.0.0.1 by default and ships with
  authentication disabled. Anyone deploying it on a network must enable the
  API-key or OIDC middleware documented in `docs/enterprise/security.md`.
- Model weights are loaded through Hugging Face Transformers with
  `trust_remote_code` disabled by default. Keep it disabled for untrusted
  model ids.
