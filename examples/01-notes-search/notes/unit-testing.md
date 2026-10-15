---
tags: [testing]
---
# Writing useful unit tests

## What to test

Test behaviour at module boundaries, not private helpers. One assertion idea per test keeps failures readable.

## Fakes

Prefer in-memory fakes over deep mocks; they break less when internals change.
