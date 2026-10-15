---
tags: [observability, devops]
---
# Structured logging

## JSON logs

Emit logs as JSON with a request id, user id and service name so they can be queried instead of grepped.

## Investigating incidents

When something breaks in prod, filter by request id to see the full story of a single failing call across services.
