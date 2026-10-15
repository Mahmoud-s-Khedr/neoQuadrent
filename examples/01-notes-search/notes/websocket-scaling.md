---
tags: [backend, scaling]
---
# Scaling WebSocket connections

## Sticky

Long-lived sockets pin clients to instances; use a pub/sub bus like Redis to fan out messages across nodes.

## Limits

Tune file descriptor limits and heartbeat intervals.
