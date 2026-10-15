---
tags: [backend, performance, nodejs]
---
# Non-blocking I/O in Node.js

## Event loop

Synchronous file reads or CPU-heavy loops block the event loop and stall every concurrent request.

## Fixes

Use async APIs, move CPU work to worker threads, and watch event loop lag as a metric.
