# Security

**Development credentials.** `docker-compose.yml` contains throwaway passwords for local containers only. They are not used anywhere else and must never be reused.

**Secrets (target design).** Application secrets (DB credentials, JWT signing key) resolve from HashiCorp Vault at startup; `.env` holds only the Vault root token and ports. The API refuses to start if Vault is unreachable. `grep -ri 'password' app/` should return nothing outside Vault-reading code.

**Model weights.** Loaded with `torch.load(weights_only=True)` after SHA-256 verification against the model card, so a tampered or swapped weights file is rejected before any code runs.

**Uploaded documents.** Treated as untrusted input: only the first page is decoded, as an image, and no OCR or text extraction is performed.

**Reporting.** Please open a private security advisory on this repository rather than a public issue.
