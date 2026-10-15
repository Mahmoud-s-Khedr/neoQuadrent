---
tags: [database, performance, postgres]
---
# Choosing database indexes

## Symptoms

Slow endpoints often come down to sequential scans. `EXPLAIN ANALYZE` shows whether PostgreSQL reads the whole table to find a handful of rows.

## Fix

Add a B-tree index on the columns used in WHERE and JOIN clauses, in the order of selectivity. Composite indexes help queries that filter on several columns at once.

## Cost

Every index slows inserts and takes disk space, so drop indexes that `pg_stat_user_indexes` shows are never used.
