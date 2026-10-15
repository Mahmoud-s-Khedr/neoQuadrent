---
tags: [observability, devops]
---
# Distributed tracing with OpenTelemetry

## Spans

Each hop between services becomes a span with timing and attributes. A trace shows exactly which dependency failed or stalled.

## Setup

Instrument with the OpenTelemetry SDK, export to Jaeger or Tempo, and propagate the trace context header on every outbound call.
