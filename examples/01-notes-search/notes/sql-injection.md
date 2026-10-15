---
tags: [security, database]
---
# Preventing SQL injection

## Parameters

Always use parameterized queries; never concatenate user input into SQL strings.

## Least privilege

The application role should not own the schema or have DROP rights.
