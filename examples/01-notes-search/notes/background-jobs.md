---
tags: [backend, architecture]
---
# Background jobs and cron schedules

## Queues

Move slow work like emails and reports out of the request into a queue such as BullMQ or Sidekiq.

## Nightly tasks

Recurring work (nightly exports, cleanup at 02:00) is defined with cron expressions in the job scheduler, with locking so only one worker runs it.
