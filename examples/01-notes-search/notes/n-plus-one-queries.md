---
tags: [database, performance, orm]
---
# The N+1 query problem in ORMs

## What it is

An ORM loads a list of orders, then issues one extra query per order to fetch its customer. A page with 100 orders makes 101 round trips to the database.

## Fix

Use eager loading (`include`, `select_related`, `joinedload`) or a batched loader like DataLoader so related rows arrive in a single query.

## Detecting it

Log the number of SQL statements per request in development and fail tests that exceed a budget.
