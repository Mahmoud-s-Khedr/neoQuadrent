---
tags: [frontend, performance]
---
# Optimizing images for the web

## Formats

Serve AVIF or WebP with a JPEG fallback using `<picture>`. Resize on upload so phones never download 4000-pixel photos.

## Delivery

Put images behind a CDN with long cache headers and use `srcset` so the browser picks the right size.
