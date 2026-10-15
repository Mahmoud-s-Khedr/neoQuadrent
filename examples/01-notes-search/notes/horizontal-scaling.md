---
tags: [architecture, scaling]
---
# Scaling out with stateless services

## Stateless

Keep sessions in Redis or signed cookies so any instance can serve any request. Then adding capacity is just adding replicas.

## Autoscaling

Scale on CPU or queue depth with Kubernetes HPA; set sensible minimums to absorb sudden traffic spikes.
