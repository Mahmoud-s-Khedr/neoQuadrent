---
tags: [backend, performance, profiling]
---
# Profiling hot paths with flame graphs

## Measure first

Before optimizing anything, record a CPU profile under realistic load. Flame graphs show which functions consume the most cycles; the widest bars are where the latency comes from.

## Tools

Use `perf` or `py-spy` on Linux, `clinic flame` for Node.js, and `pprof` for Go. Sample in production at a low rate so the overhead stays negligible.

## What we found

A JSON serializer called inside a loop accounted for 40% of CPU per request. Hoisting it out made every endpoint noticeably snappier.
