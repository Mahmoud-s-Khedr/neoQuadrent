---
tags: [frontend, performance]
---
# Code splitting the JavaScript bundle

## Problem

Our single 2.3 MB bundle meant the first paint took ages on mobile networks.

## Approach

Split by route with dynamic `import()`, move rarely used libraries into separate chunks, and preload only what the first screen needs. Lighthouse went from 41 to 88.
