---
tags: [frontend, security]
---
# Understanding CORS

## Same origin

Browsers block cross-origin requests unless the target sends Access-Control-Allow-Origin.

## Preflight

Non-simple requests trigger an OPTIONS preflight; cache it with Access-Control-Max-Age.
