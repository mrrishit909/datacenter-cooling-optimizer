# Evaluation

Five halls the demo never uses (seeds 21, 31, 41, 51, 61). Each has its own rack loads, CRAC capacities, recirculation,
airflow knee and sensor offsets. The model is fitted to the readings only; every outcome below is measured in the simulated hall.
Reproduce with `python -m dc.evaluate`.

## The thermal model on ordinary days

Fitted on days 1–50, checked on days 51–60 (all CRACs running). Error against the readings and against the true inlet temperature.

| Hall | MAE vs readings | 99th pct | MAE vs true inlet | Design rule (supply + 1.5 °C) MAE | Unit-stop windows held out: MAE, worst under-prediction |
|---|---|---|---|---|---|
| 21 | 0.22 °C | 0.73 °C | 0.29 °C | 0.86 °C | 0.24/1.0, 0.30/1.3, 0.30/0.7, 0.31/1.1 |
| 31 | 0.22 °C | 0.71 °C | 0.34 °C | 0.77 °C | 0.33/1.4, 0.25/1.1, 0.31/1.2, 0.33/1.1 |
| 41 | 0.22 °C | 0.72 °C | 0.31 °C | 0.75 °C | 0.39/0.8, 0.22/0.7, 0.21/0.8, 0.39/0.6 |
| 51 | 0.22 °C | 0.72 °C | 0.30 °C | 0.72 °C | 0.36/1.2, 0.24/1.1, 0.25/1.1, 0.27/0.9 |
| 61 | 0.22 °C | 0.71 °C | 0.32 °C | 0.76 °C | 0.29/1.0, 0.34/1.2, 0.24/1.1, 0.23/0.8 |

## The surge: 24 GPU servers (240 kW) placed two ways

Predicted inlets at the forecast peak hour against the true inlets at that hour, at 18 °C everywhere; then the set-points the cooling
LP chooses (margin doubled because the surge is outside the fitted load) and the simulated day at those set-points.

| Hall | Placement | Predicted max | True max | Rack MAE | Worst under-prediction | Set-points | True max, planned | Racks over 27 °C | Cooling kWh/day (18 °C) | PUE (18 °C) |
|---|---|---|---|---|---|---|---|---|---|---|
| 21 | First fit | 26.3 | 25.1 | 0.52 | 1.4 | 16 / 19.5 / 23 / 22.5 | 25.0 | 0 | 6,735 (7,277) | 1.444 (1.475) |
| 21 | Thermal-aware | 22.2 | 22.8 | 0.53 | 1.5 | 21 / 21.5 / 22.5 / 20.5 | 26.1 | 0 | 5,631 (7,277) | 1.381 (1.475) |
| 31 | First fit | 24.2 | 23.4 | 0.52 | 2.4 | 18.5 / 22.5 / 22 / 23 | 25.5 | 0 | 5,469 (6,883) | 1.392 (1.477) |
| 31 | Thermal-aware | 21.9 | 21.9 | 0.67 | 2.2 | 21.5 / 22 / 21 / 22.5 | 25.9 | 0 | 5,192 (6,883) | 1.375 (1.477) |
| 41 | First fit | 24.1 | 23.8 | 0.61 | 2.3 | 19 / 19.5 / 22.5 / 23.5 | 25.4 | 0 | 5,814 (6,994) | 1.406 (1.477) |
| 41 | Thermal-aware | 22.0 | 22.9 | 0.70 | 2.9 | 23 / 21 / 21.5 / 22 | 26.6 | 0 | 5,242 (6,994) | 1.372 (1.477) |
| 51 | First fit | 23.7 | 23.3 | 0.67 | 0.8 | 19 / 20 / 23 / 23.5 | 25.1 | 0 | 5,393 (6,601) | 1.403 (1.480) |
| 51 | Thermal-aware | 20.8 | 21.6 | 0.39 | 1.5 | 22.5 / 22.5 / 22.5 / 23 | 26.2 | 0 | 4,698 (6,601) | 1.359 (1.480) |
| 61 | First fit | 23.4 | 23.6 | 0.64 | 3.0 | 17.5 / 22 / 23 / 23 | 26.2 | 0 | 6,162 (7,547) | 1.397 (1.473) |
| 61 | Thermal-aware | 21.4 | 22.1 | 0.54 | 1.8 | 22 / 22 / 22 / 22 | 26.1 | 0 | 5,570 (7,547) | 1.365 (1.473) |

Cooling energy saved against fixed 18 °C: thermal-aware 23%–29%, first fit 7%–21%.

## One CRAC stops at the surge load

Each of the four units stopped in turn, after each placement and its cooling plan. *Model said*: the hottest inlet the model predicts
with no action. *Runbook*: every running unit to 16 °C. *Open loop*: the model's re-plan of set-points and GPU power caps with the
airflow rule, without reading the sensors. *Closed loop* (what the API does): runbook, read the sensors ten minutes later, correct the
model with them, re-plan. Racks over 27 °C at any hour of the day / GPU kW capped.

| Hall | Placement | Unit | Model said | No action: true max, racks over | Runbook | Open loop | Closed loop |
|---|---|---|---|---|---|---|---|
| 21 | First fit | CRAC-1 | 30.0 | 47.9, 24 | 43.0 °C, 21 over, 0 kW capped | infeasible | review, fallback: 25.0 °C, 0 over, 189 kW capped |
| 21 | First fit | CRAC-2 | 26.7 | 44.3, 24 | 41.7 °C, 23 over, 0 kW capped | infeasible | review, fallback: 24.4 °C, 0 over, 189 kW capped |
| 21 | First fit | CRAC-3 | 25.4 | 24.7, 0 | 23.4 °C, 0 over, 0 kW capped | 25.3 °C, 0 over, 54 kW capped | 26.0 °C, 0 over, 0 kW capped |
| 21 | First fit | CRAC-4 | 25.3 | 24.8, 0 | 23.4 °C, 0 over, 0 kW capped | 25.0 °C, 0 over, 55 kW capped | 26.1 °C, 0 over, 0 kW capped |
| 21 | Thermal-aware | CRAC-1 | 26.0 | 32.4, 19 | 26.6 °C, 0 over, 0 kW capped | 25.8 °C, 0 over, 88 kW capped | 24.8 °C, 0 over, 13 kW capped |
| 21 | Thermal-aware | CRAC-2 | 25.9 | 31.6, 23 | 26.7 °C, 0 over, 0 kW capped | 25.6 °C, 0 over, 96 kW capped | 24.8 °C, 0 over, 14 kW capped |
| 21 | Thermal-aware | CRAC-3 | 25.5 | 34.5, 22 | 29.8 °C, 15 over, 0 kW capped | infeasible | 25.1 °C, 0 over, 48 kW capped |
| 21 | Thermal-aware | CRAC-4 | 26.2 | 36.9, 24 | 30.9 °C, 14 over, 0 kW capped | infeasible | 24.7 °C, 0 over, 62 kW capped |
| 31 | First fit | CRAC-1 | 28.8 | 41.7, 24 | 35.4 °C, 18 over, 0 kW capped | infeasible | 24.2 °C, 0 over, 127 kW capped |
| 31 | First fit | CRAC-2 | 25.2 | 34.8, 22 | 30.5 °C, 16 over, 0 kW capped | infeasible | 24.9 °C, 0 over, 84 kW capped |
| 31 | First fit | CRAC-3 | 25.5 | 25.5, 0 | 21.5 °C, 0 over, 0 kW capped | 24.7 °C, 0 over, 10 kW capped | 26.2 °C, 0 over, 0 kW capped |
| 31 | First fit | CRAC-4 | 25.4 | 25.4, 0 | 21.5 °C, 0 over, 0 kW capped | 25.1 °C, 0 over, 18 kW capped | 25.3 °C, 0 over, 0 kW capped |
| 31 | Thermal-aware | CRAC-1 | 25.9 | 30.5, 23 | 24.8 °C, 0 over, 0 kW capped | 24.8 °C, 0 over, 96 kW capped | 24.8 °C, 0 over, 0 kW capped |
| 31 | Thermal-aware | CRAC-2 | 25.3 | 29.5, 12 | 23.7 °C, 0 over, 0 kW capped | 25.0 °C, 0 over, 80 kW capped | 24.0 °C, 0 over, 0 kW capped |
| 31 | Thermal-aware | CRAC-3 | 26.5 | 32.6, 22 | 26.4 °C, 0 over, 0 kW capped | infeasible | 24.9 °C, 0 over, 12 kW capped |
| 31 | Thermal-aware | CRAC-4 | 25.3 | 31.7, 19 | 26.4 °C, 0 over, 0 kW capped | infeasible | 24.6 °C, 0 over, 20 kW capped |
| 41 | First fit | CRAC-1 | 26.9 | 40.2, 23 | 35.4 °C, 18 over, 0 kW capped | infeasible | 25.0 °C, 0 over, 134 kW capped |
| 41 | First fit | CRAC-2 | 26.7 | 40.7, 24 | 35.8 °C, 21 over, 0 kW capped | infeasible | 24.9 °C, 0 over, 142 kW capped |
| 41 | First fit | CRAC-3 | 25.4 | 25.5, 0 | 21.8 °C, 0 over, 0 kW capped | 24.5 °C, 0 over, 29 kW capped | 26.0 °C, 0 over, 0 kW capped |
| 41 | First fit | CRAC-4 | 25.3 | 25.3, 0 | 21.8 °C, 0 over, 0 kW capped | 24.8 °C, 0 over, 29 kW capped | 22.3 °C, 0 over, 0 kW capped |
| 41 | Thermal-aware | CRAC-1 | 25.3 | 28.2, 14 | 23.1 °C, 0 over, 0 kW capped | 25.4 °C, 0 over, 74 kW capped | 24.4 °C, 0 over, 0 kW capped |
| 41 | Thermal-aware | CRAC-2 | 26.1 | 31.6, 22 | 25.1 °C, 0 over, 0 kW capped | 25.2 °C, 0 over, 89 kW capped | 24.9 °C, 0 over, 1 kW capped |
| 41 | Thermal-aware | CRAC-3 | 25.6 | 35.5, 24 | 29.2 °C, 8 over, 0 kW capped | infeasible | 25.7 °C, 0 over, 40 kW capped |
| 41 | Thermal-aware | CRAC-4 | 25.5 | 31.5, 24 | 26.0 °C, 0 over, 0 kW capped | 25.3 °C, 0 over, 108 kW capped | 25.0 °C, 0 over, 13 kW capped |
| 51 | First fit | CRAC-1 | 28.0 | 43.6, 22 | 38.2 °C, 19 over, 0 kW capped | infeasible | 24.2 °C, 0 over, 136 kW capped |
| 51 | First fit | CRAC-2 | 26.6 | 40.7, 24 | 35.6 °C, 22 over, 0 kW capped | infeasible | 24.2 °C, 0 over, 128 kW capped |
| 51 | First fit | CRAC-3 | 25.4 | 25.1, 0 | 21.3 °C, 0 over, 0 kW capped | 25.1 °C, 0 over, 28 kW capped | 26.0 °C, 0 over, 0 kW capped |
| 51 | First fit | CRAC-4 | 25.4 | 24.7, 0 | 21.3 °C, 0 over, 0 kW capped | 24.9 °C, 0 over, 28 kW capped | 25.4 °C, 0 over, 0 kW capped |
| 51 | Thermal-aware | CRAC-1 | 25.9 | 32.6, 23 | 26.0 °C, 0 over, 0 kW capped | 25.7 °C, 0 over, 102 kW capped | 24.9 °C, 0 over, 9 kW capped |
| 51 | Thermal-aware | CRAC-2 | 25.5 | 33.3, 24 | 26.6 °C, 0 over, 0 kW capped | infeasible | 24.9 °C, 0 over, 14 kW capped |
| 51 | Thermal-aware | CRAC-3 | 25.9 | 31.6, 22 | 24.8 °C, 0 over, 0 kW capped | 25.4 °C, 0 over, 82 kW capped | 25.0 °C, 0 over, 0 kW capped |
| 51 | Thermal-aware | CRAC-4 | 25.4 | 30.6, 22 | 24.1 °C, 0 over, 0 kW capped | 25.6 °C, 0 over, 84 kW capped | 24.4 °C, 0 over, 0 kW capped |
| 61 | First fit | CRAC-1 | 28.4 | 47.3, 24 | 40.9 °C, 21 over, 0 kW capped | infeasible | review, fallback: 25.3 °C, 0 over, 184 kW capped |
| 61 | First fit | CRAC-2 | 25.5 | 43.3, 23 | 39.5 °C, 22 over, 0 kW capped | infeasible | 24.9 °C, 0 over, 182 kW capped |
| 61 | First fit | CRAC-3 | 25.2 | 27.3, 3 | 22.9 °C, 0 over, 0 kW capped | 25.2 °C, 0 over, 82 kW capped | 26.6 °C, 0 over, 0 kW capped |
| 61 | First fit | CRAC-4 | 25.3 | 27.3, 3 | 23.0 °C, 0 over, 0 kW capped | 24.7 °C, 0 over, 86 kW capped | 25.8 °C, 0 over, 0 kW capped |
| 61 | Thermal-aware | CRAC-1 | 25.6 | 34.0, 23 | 28.0 °C, 5 over, 0 kW capped | infeasible | 24.6 °C, 0 over, 34 kW capped |
| 61 | Thermal-aware | CRAC-2 | 25.8 | 33.8, 23 | 27.8 °C, 5 over, 0 kW capped | infeasible | 25.1 °C, 0 over, 27 kW capped |
| 61 | Thermal-aware | CRAC-3 | 25.4 | 33.6, 24 | 27.6 °C, 2 over, 0 kW capped | infeasible | 24.8 °C, 0 over, 28 kW capped |
| 61 | Thermal-aware | CRAC-4 | 25.5 | 35.9, 22 | 29.9 °C, 11 over, 0 kW capped | infeasible | 24.9 °C, 0 over, 53 kW capped |

## Summary over the 40 failures

| Placement | Response | Failures with any rack over 27 °C | No plan found | GPU kW capped (mean of plans) | Cooling kWh/day (mean) |
|---|---|---|---|---|---|
| Thermal-aware | No action | 20 of 20 | 0 | 0 | 5,086 |
| Thermal-aware | Runbook, 16 °C | 7 of 20 | 0 | 0 | 8,211 |
| Thermal-aware | Open-loop model re-plan | 0 of 10 | 10 | 90 | 4,535 |
| Thermal-aware | Closed loop (runbook, read, re-plan; fallback when no plan) | 0 of 20 | 0 | 19 | 7,859 |
| First fit | No action | 12 of 20 | 0 | 0 | 5,725 |
| First fit | Runbook, 16 °C | 10 of 20 | 0 | 0 | 8,211 |
| First fit | Open-loop model re-plan | 0 of 10 | 10 | 42 | 5,355 |
| First fit | Closed loop (runbook, read, re-plan; fallback when no plan) | 0 of 20 | 3 | 75 | 6,334 |

In the 32 failures that put a rack over 27 °C, the model's no-action prediction said so in 4, and was 2.0–18.9 °C
below the true hottest inlet: it has only seen units stopped at about half this load, when the remaining air was enough. It is not trusted
there; the closed loop acts on the readings.

Run time 9 s on one core.
