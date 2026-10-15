---
tags: [database, deploy]
---
# Zero-downtime schema migrations

## Expand and contract

Add new columns as nullable, backfill, switch reads and writes, and only then drop the old column in a later release.

## Locks

Avoid long table locks: create indexes concurrently and batch large updates.
