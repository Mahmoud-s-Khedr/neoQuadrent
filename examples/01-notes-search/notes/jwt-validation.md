---
tags: [security, auth]
---
# Validating JWTs correctly

## Checks

Verify the signature with the expected algorithm, then check `exp`, `nbf`, `iss` and `aud`. Reject `alg: none` and never let the header pick the algorithm.

## Rotation

Publish keys via JWKS and cache them; refresh when an unknown `kid` arrives.
