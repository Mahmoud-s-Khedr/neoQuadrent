---
tags: [frontend, performance]
---
# Lazy loading below-the-fold content

## Idea

Do not fetch what the visitor cannot see yet. Native `loading="lazy"` on images and iframes defers them until they scroll near the viewport.

## Components

Heavy widgets like charts and maps can mount on intersection, keeping the initial page light.
