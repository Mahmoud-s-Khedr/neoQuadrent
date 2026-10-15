---
tags: [backend, payments]
---
# Idempotency keys for payment APIs

## Duplicates

Network retries can submit the same charge twice. The client sends a unique Idempotency-Key header with each logical operation.

## Storage

Store the key with the result; when the same key arrives again, return the stored result instead of charging the card again.
