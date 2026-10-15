---
tags: [security, auth]
---
# Multi-factor authentication with TOTP

## Why

A stolen or guessed secret alone should not be enough to get into an account. A second factor such as a TOTP app or a hardware key blocks most account takeovers.

## Setup

Generate a per-user seed, show it as a QR code, and verify a six-digit code before enabling. Provide one-time recovery codes.
