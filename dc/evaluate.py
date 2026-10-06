"""Held-out evaluation: halls the demo never uses. For each, the thermal model is fitted to its sixty days of sensor readings exactly
as the API fits it, the same GPU surge is placed two ways, the set-points are planned, and then each of the four CRACs is stopped in
turn; every plan is scored in the simulated hall, whose true physics no model sees.

    python -m dc.evaluate > docs/evaluation.md
"""
import time

import numpy as np

from . import engine as E, world as W

SEEDS = (21, 31, 41, 51, 61)


def rng_reading(rm, cpu, gpu, sp, on, r):
    t = W.inlet(rm, cpu, gpu, sp, on)[0]
    return t + rm["bias"] + r.normal(0, 0.25, W.N_RACKS), cpu * (1 + 0.01 * r.standard_normal(W.N_RACKS)), gpu * (1 + 0.01 * r.standard_normal(W.N_RACKS))


def failure(m, inv, rm, cpu_plan, cpu_day, ph, g, sp, k, r):
    """One CRAC stops: no action, the runbook, the model's open-loop re-plan, and the closed loop (runbook, read, correct, re-plan)."""
    on = np.ones(4, bool)
    on[k] = False
    out = {"said": float(E.predicted(m, cpu_plan, g, sp, on).max()), "none": E.score(rm, cpu_day, g, sp, on)}
    rb = np.where(on, W.SET_MIN, np.nan)
    out["runbook"] = E.score(rm, cpu_day, g, np.nan_to_num(rb, nan=18.0), on)
    op = E.replan(m, inv, cpu_plan, g, on, E.margin(m, True, True), rule=True)
    out["open"] = None if op is None else E.score(rm, cpu_day, g, np.nan_to_num(op[0], nan=18.0), on, op[1])
    read, cr, gr = rng_reading(rm, cpu_day[ph], g * 0.95, np.nan_to_num(rb, nan=18.0), on, r)
    offset, kappa, use = E.correction(m, inv, cr, gr, rb, on, read)
    cl = E.replan(m, inv, cpu_plan, g, on, m["margin_outage"], rule=False, offset=offset, kappa=kappa)
    if cl is None:      # the API's fallback: runbook set-points, GPUs in the starved aisle at the lowest cap, and a review task
        cl = (rb, np.where((g > 0) & (use[inv["zone"]] > E.RULE), E.CAP_MIN, 1.0))
        out["review"] = True
    out["closed"] = E.score(rm, cpu_day, g, np.nan_to_num(cl[0], nan=18.0), on, cl[1])
    return out


def main():
    t0 = time.time()
    fits, places, fails = [], [], []
    for seed in SEEDS:
        h = W.history(seed)
        rm = h["room"]
        d = E.dataset(h)
        m, ev = E.calibrate(d, rm["zone"], h["inlet_true"])
        fits.append((seed, ev, m))
        fc = E.heat_forecast(d["cpu"])
        ph = int(fc.sum(1).argmax())
        cpu_plan = fc[ph]
        cpu_day, _ = W.day_ahead(rm, seed)
        inv = {"zone": rm["zone"], "row": rm["row"], "free": rm["free"], "gpu": rm["gpu"], "caps": rm["crac_cap"]}
        on = np.ones(4, bool)
        sp18 = np.full(4, 18.0)
        ff, _ = W.first_fit(rm)
        ta, _, _ = E.place_thermal(m, inv, cpu_plan, sp18, W.SURGE)
        r = np.random.default_rng(seed + 3)
        for name, add in (("First fit", ff), ("Thermal-aware", ta)):
            g = (rm["gpu"] + add) * W.GPU_KW
            pred = E.predicted(m, cpu_plan, g, sp18)
            at18 = E.score(rm, cpu_day, g, sp18, on)
            true_peak = at18["inlet"][ph]
            xp, _ = E.extrapolating(m, cpu_plan, g, rm["zone"])
            sp = E.cooling_plan(m, cpu_plan, g, on, E.margin(m, xp))
            planned = E.score(rm, cpu_day, g, sp, on) if sp is not None else None
            places.append((seed, name, pred, true_peak, at18, sp, planned, xp))
            for k in range(4):
                fails.append((seed, name, k, failure(m, inv, rm, cpu_plan, cpu_day, ph, g, sp if sp is not None else sp18, k, r)))
    out = ["# Evaluation", "", f"Five halls the demo never uses (seeds {', '.join(map(str, SEEDS))}). Each has its own rack loads, CRAC capacities, recirculation,",
           "airflow knee and sensor offsets. The model is fitted to the readings only; every outcome below is measured in the simulated hall.",
           "Reproduce with `python -m dc.evaluate`.", "",
           "## The thermal model on ordinary days", "",
           "Fitted on days 1–50, checked on days 51–60 (all CRACs running). Error against the readings and against the true inlet temperature.", "",
           "| Hall | MAE vs readings | 99th pct | MAE vs true inlet | Design rule (supply + 1.5 °C) MAE | Unit-stop windows held out: MAE, worst under-prediction |",
           "|---|---|---|---|---|---|"]
    for seed, ev, m in fits:
        oh = ", ".join(f"{o['mae']:.2f}/{o['worst_under']:.1f}" for o in ev["outage_holdout"])
        out.append(f"| {seed} | {ev['val_mae']:.2f} °C | {ev['val_p99']:.2f} °C | {ev['truth_val_mae']:.2f} °C | {ev['design_rule_mae']:.2f} °C | {oh} |")
    out += ["", "## The surge: 24 GPU servers (240 kW) placed two ways", "",
            "Predicted inlets at the forecast peak hour against the true inlets at that hour, at 18 °C everywhere; then the set-points the cooling",
            "LP chooses (margin doubled because the surge is outside the fitted load) and the simulated day at those set-points.", "",
            "| Hall | Placement | Predicted max | True max | Rack MAE | Worst under-prediction | Set-points | True max, planned | Racks over 27 °C | Cooling kWh/day (18 °C) | PUE (18 °C) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for seed, name, pred, tp, at18, sp, pl, xp in places:
        sps = " / ".join(f"{x:g}" for x in sp) if sp is not None else "infeasible"
        out.append(f"| {seed} | {name} | {pred.max():.1f} | {tp.max():.1f} | {np.abs(pred - tp).mean():.2f} | {(tp - pred).max():.1f} | {sps} | "
                   f"{pl['max_inlet']:.1f} | {pl['racks_over_limit']} | {pl['cooling_kwh']:,.0f} ({at18['cooling_kwh']:,.0f}) | {pl['pue']:.3f} ({at18['pue']:.3f}) |")
    ta = [p for p in places if p[1] == "Thermal-aware"]
    ff = [p for p in places if p[1] == "First fit"]
    save_ta = [1 - p[6]["cooling_kwh"] / p[4]["cooling_kwh"] for p in ta]
    save_ff = [1 - p[6]["cooling_kwh"] / p[4]["cooling_kwh"] for p in ff]
    out += ["", f"Cooling energy saved against fixed 18 °C: thermal-aware {min(save_ta):.0%}–{max(save_ta):.0%}, first fit {min(save_ff):.0%}–{max(save_ff):.0%}.", "",
            "## One CRAC stops at the surge load", "",
            "Each of the four units stopped in turn, after each placement and its cooling plan. *Model said*: the hottest inlet the model predicts",
            "with no action. *Runbook*: every running unit to 16 °C. *Open loop*: the model's re-plan of set-points and GPU power caps with the",
            "airflow rule, without reading the sensors. *Closed loop* (what the API does): runbook, read the sensors ten minutes later, correct the",
            "model with them, re-plan. Racks over 27 °C at any hour of the day / GPU kW capped.", "",
            "| Hall | Placement | Unit | Model said | No action: true max, racks over | Runbook | Open loop | Closed loop |", "|---|---|---|---|---|---|---|---|"]
    for seed, name, k, f in fails:
        cell = lambda s: "infeasible" if s is None else f"{s['max_inlet']:.1f} °C, {s['racks_over_limit']} over, {s['gpu_kw_capped']:.0f} kW capped"     # noqa: E731
        out.append(f"| {seed} | {name} | {W.CRACS[k][0]} | {f['said']:.1f} | {f['none']['max_inlet']:.1f}, {f['none']['racks_over_limit']} | {cell(f['runbook'])} | {cell(f['open'])} | "
                   f"{'review, fallback: ' if f.get('review') else ''}{cell(f['closed'])} |")
    out += ["", "## Summary over the 40 failures", "", "| Placement | Response | Failures with any rack over 27 °C | No plan found | GPU kW capped (mean of plans) | Cooling kWh/day (mean) |", "|---|---|---|---|---|---|"]
    for place in ("Thermal-aware", "First fit"):
        for key, label in (("none", "No action"), ("runbook", "Runbook, 16 °C"), ("open", "Open-loop model re-plan"), ("closed", "Closed loop (runbook, read, re-plan; fallback when no plan)")):
            rs = [f[key] for _, n, _, f in fails if n == place]
            ok = [s for s in rs if s is not None]
            nope = sum(s is None for s in rs) if key != "closed" else sum(bool(f.get("review")) for _, n, _, f in fails if n == place)
            out.append(f"| {place} | {label} | {sum(s['racks_over_limit'] > 0 for s in ok)} of {len(ok)} | {nope} | {np.mean([s['gpu_kw_capped'] for s in ok]):.0f} | {np.mean([s['cooling_kwh'] for s in ok]):,.0f} |")
    bad = [f for *_, f in fails if f["none"]["racks_over_limit"]]
    under = [f["none"]["max_inlet"] - f["said"] for f in bad]
    out += ["", f"In the {len(bad)} failures that put a rack over 27 °C, the model's no-action prediction said so in {sum(f['said'] > W.LIMIT_C for f in bad)}, and was {min(under):.1f}–{max(under):.1f} °C",
            "below the true hottest inlet: it has only seen units stopped at about half this load, when the remaining air was enough. It is not trusted",
            "there; the closed loop acts on the readings.", "",
            f"Run time {time.time() - t0:.0f} s on one core."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
