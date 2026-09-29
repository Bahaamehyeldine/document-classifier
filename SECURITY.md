# Security

**Secrets.** Application secrets (database URL, JWT signing key, MinIO and SFTP credentials) are read from HashiCorp Vault at startup by `app/infra/vault.py`; `.env` holds only the Vault root token and ports. The api, worker and sftp-ingest refuse to start if Vault is unreachable or a secret is missing. `Secrets.__repr__` is redacted so secrets can't leak into logs.

**`grep -ri password app/`** returns only identifiers, never values:
- `app/infra/vault.py`: the `sftp_password` key read from Vault (the Vault-reading code),
- `app/workers/sftp_ingest.py`: passing that Vault value to the SFTP adapter,
- `app/infra/sftp.py`: paramiko's `password=` keyword argument,
- `app/auth.py`: fastapi-users' `reset_password_token_secret` attribute (set from the Vault JWT key),
- `app/cli.py`: fastapi-users' `password` field when creating the first admin.

`tests/unit/test_architecture.py::test_no_hard_coded_secrets` fails CI if any password, secret, token or API-key name in `app/` is assigned a quoted literal.

**Development credentials.** `docker-compose.yml` contains throwaway credentials for the local containers (Postgres, MinIO, SFTP) and seeds them into dev-mode Vault. They exist only in the local stack and must not be reused. Dev-mode Vault is unsealed and in-memory; a real deployment would use a sealed Vault with AppRole or Kubernetes auth instead of the root token.

**Authentication and authorization.** JWT bearer tokens (fastapi-users, bcrypt/argon2 password hashing). Every endpoint except health, register and login requires a token, and every data endpoint checks a Casbin permission. New accounts have no permissions until an admin grants a role. The last admin cannot be removed.

**Audit trail.** Role changes, invitations, relabels and batch state changes are written to an append-only `audit_log` table with actor, target, details, request id and timestamp. The api exposes no update or delete for it.

**Model weights.** Verified by SHA-256 against the model card, then loaded with `torch.load(weights_only=True)`, so a tampered or swapped file is rejected before any of its code could run.

**Uploaded documents.** Treated as untrusted input: only the first page is decoded, as an image, with no OCR or text extraction. Files that fail to decode are marked failed, not retried.

**Containers.** The application image runs as an unprivileged user.

**Reporting.** Please open a private security advisory on this repository rather than a public issue.
