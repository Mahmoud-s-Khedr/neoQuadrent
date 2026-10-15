---
tags: [database, scaling]
---
# Database connection pooling

## Why

Opening a PostgreSQL connection is expensive and each one costs memory. Under heavy concurrency the database runs out of slots.

## PgBouncer

Put PgBouncer in transaction mode between the app and the database so thousands of clients share a few dozen connections.
