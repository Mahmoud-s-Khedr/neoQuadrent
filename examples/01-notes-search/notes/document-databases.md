---
tags: [database, architecture]
---
# When a document database fits

## Flexible shape

MongoDB and similar stores keep records as JSON documents, so fields can vary between records and evolve without migrations.

## Trade-offs

Joins and multi-document transactions are weaker; model data around how it is read.
