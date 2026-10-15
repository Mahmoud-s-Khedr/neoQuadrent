---
tags: [security, auth]
---
# Rate limiting the login endpoint

## Brute force

Attackers try thousands of combinations against sign-in forms. Limit attempts per account and per IP address with a sliding window.

## Implementation

A token bucket in Redis works well. After five failures, add an exponential delay and show a CAPTCHA.
