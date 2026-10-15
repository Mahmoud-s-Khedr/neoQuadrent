---
tags: [security, devops]
---
# Managing API keys and other secrets

## Do not commit them

Tokens and keys in git history leak eventually. Scan repositories with gitleaks and rotate anything that was ever pushed.

## Where to keep them

Use a vault (HashiCorp Vault, AWS Secrets Manager, Doppler) and inject values as environment variables at deploy. Give each service its own narrowly scoped credentials.
