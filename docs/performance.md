# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 48 racks, 4 CRACs, 152 sensors, 218,524 hourly readings,
51 GPU servers, five decisions. Reproduce with `make demo`, then `make load-test`.

## Reads: the hall's PUE and the hall overview, closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 92 | 9.1 req/s | 121.5 ms | 147.7 ms | 196.4 ms | 0.00% |
| 8 | 373 | 36.7 req/s | 203.5 ms | 270.4 ms | 320.5 ms | 0.00% |
| 32 | 301 | 27.2 req/s | 976.1 ms | 1181.4 ms | 1244.3 ms | 0.00% |

Slow for reads, and the reason is plain: the PUE call reads fourteen days of telemetry (about 51,000 rows) on every request to rebuild
the heat forecast and the week's PUE, then simulates the next day. Under 300 ms at p95 up to eight concurrent clients; past that one
process saturates. The fix, not built: keep the forecast and the hourly aggregates in a materialised table refreshed when telemetry
arrives, which turns each read into a few hundred rows.

## Compute (in-process, one core)

| Work | Time |
|---|---|
| Generate the hall and sixty days of hourly operation | 0.01 s |
| Load job: racks, servers, sensors and 218,524 readings into Postgres | most of the demo's 5 s |
| Fit the surrogate (48 bounded ridge fits), ten-day check and four held-out maintenance refits | 1.9 s |
| Placement MILP: 48 integers, 24 binaries, 50 constraints | 0.03 s |
| Cooling-plan LP and failure re-plan LP | under 5 ms each |
| A day in the simulated hall (24 hours x 48 racks) | about 1 ms |
| The whole demo, six steps | about 5 s |
| Evaluation: five halls, two placements, four failures each | about 9 s |

Not measured: halls of thousands of racks (the surrogate is per rack over its aisle, so it grows with aisle length, not hall size; the
MILP grows with racks x rows), or sub-minute telemetry, which belongs in a time-series store, not this table.
