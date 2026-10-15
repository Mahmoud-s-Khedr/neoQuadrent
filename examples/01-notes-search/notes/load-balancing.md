---
tags: [architecture, scaling]
---
# Load balancing strategies

## Algorithms

Round robin is fine for uniform requests; least-connections works better when request costs vary. Health checks remove broken instances automatically.

## Layer

Layer 7 balancers (nginx, Envoy) can route by path and terminate TLS.
