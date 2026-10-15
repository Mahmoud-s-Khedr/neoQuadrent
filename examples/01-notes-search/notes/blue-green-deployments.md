---
tags: [devops, deploy]
---
# Blue-green deployments

## How

Run two identical environments. Deploy the new version to the idle one, run smoke checks, then switch the load balancer.

## Undo

If errors spike after the switch, point traffic back at the previous environment in seconds. No rebuild is needed.
