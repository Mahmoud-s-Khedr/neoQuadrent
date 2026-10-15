---
tags: [backend, performance, caching]
---
# Adding a Redis cache in front of the API

## Why cache

Most API endpoints read the same rows again and again. Keeping hot results in Redis lets the backend answer from memory instead of hitting PostgreSQL, which cut our p95 latency from 480 ms to 60 ms.

## Invalidation

Use short TTLs for data that changes often and explicit key deletion on writes. Cache-aside is the simplest pattern: read from cache, fall back to the database, then populate the cache.

## Pitfalls

Watch for cache stampedes when a popular key expires; add jitter to TTLs or use a lock so only one request rebuilds the entry.
