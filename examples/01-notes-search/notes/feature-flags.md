---
tags: [devops, deploy]
---
# Shipping behind feature flags

## Decouple

Merge code dark and enable it with a flag per user segment. A misbehaving feature can be switched off instantly without a new release.

## Hygiene

Delete flags once a rollout completes; stale flags become technical debt.
