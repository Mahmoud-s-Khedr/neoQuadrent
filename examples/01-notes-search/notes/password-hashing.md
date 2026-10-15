---
tags: [security, auth]
---
# Storing credentials with Argon2

## Never plain text

Credentials must never be persisted as plain text or with fast digests like MD5 or SHA-256. Use a slow, salted, memory-hard function.

## Argon2id

Argon2id with a unique random salt per user is the current recommendation. Tune memory and iterations so hashing takes roughly 200 ms on your hardware.

## Migration

Rehash legacy digests on the next successful login and mark the account as migrated.
