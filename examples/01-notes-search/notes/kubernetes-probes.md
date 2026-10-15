---
tags: [devops, kubernetes]
---
# Kubernetes liveness and readiness probes

## Readiness

Readiness gates traffic until the pod can serve; liveness restarts a stuck container.

## Careful

Do not point liveness at dependencies, or a database blip restarts the whole fleet.
