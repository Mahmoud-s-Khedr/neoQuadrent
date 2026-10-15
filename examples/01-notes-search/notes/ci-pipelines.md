---
tags: [devops, testing]
---
# Continuous integration pipelines

## Gate merges

Every pull request runs lint, type checks and the test suite in GitHub Actions; branch protection blocks merging until all checks pass.

## Speed

Cache dependencies and split tests across parallel jobs to keep feedback under ten minutes.
