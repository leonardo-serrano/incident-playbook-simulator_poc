## CPU Saturation (DB)
If CPU > 90% for more than 5m:
1. Check logs for slow queries.
2. If active connections exceed threshold, scale instances by +1.
3. If issue persists after scaling, escalate to DB team.

## HTTP 500 Spike (Web/API)
If p95 latency > 2s and HTTP 500 spike post-deployment:
1. Check error logs for known patterns.
2. If matches last deployment, rollback to previous version.
3. Notify incident channel after action.
