---
tags: [backend, performance]
---
# HTTP compression with Brotli

## Smaller payloads

Brotli shrinks JSON and HTML by 70-80%, so clients receive data far sooner on slow links.

## Where

Compress at the reverse proxy (nginx, Caddy) and precompress static assets at build.
