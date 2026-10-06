"""Models and optimizers. They see only what the sensors and the inventory say: rack power by PDU branch, CRAC supply temperature and
status, rack inlet readings, the racks' free slots and the CRACs' commissioning airflow. They never see dc.world's hidden physics.

Thermal surrogate: per rack, inlet = intercept + supply-air terms + heat from every rack in the same aisle, fitted by bounded ridge
regression (heat and supply terms cannot be negative). An outage flag per CRAC learns what the eight-hour maintenance windows showed.
Placement: a MILP over how many GPU servers go in each rack and which row each job lands in (a job stays in one row for its network),
minimising the hottest predicted inlet. Cooling: an LP raising the set-points as far as the predicted inlets allow. Failure re-plan:
an LP over set-points and GPU power caps. Outside the load the model was fitted on, an airflow rule from the commissioning figures
is added: an aisle's IT airflow must stay under 85% of the air its running units deliver.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, linprog, milp

from . import world as W

VERSION = "surrogate-1"
RULE = 0.85             # an aisle's IT airflow demand over the air delivered to it, the fallback rule's ceiling
CROSS = 0.1             # share of a CRAC's air that reaches the other aisle (commissioning study)
GPU_AIR = 0.8           # GPU servers move less air per kW than CPU servers (datasheet)
CAP_MIN = 0.5           # lowest GPU power cap the re-plan may set
RIDGE = 30.0
EXTRAPOLATION_FACTOR = 2.0   # outside the fitted load range the safety margin doubles
TRAIN_DAYS = 50


# ---------------------------------------------------------------- the surrogate

def supply_features(tsup, on):
    """A stopped unit's term takes the mean of the running units' supply air: its share of the air now comes from them."""
    tsup, on = np.atleast_2d(tsup).astype(float), np.atleast_2d(on).astype(bool)
    mean_on = np.nanmean(np.where(on, tsup, np.nan), 1, keepdims=True)
    return np.where(on, tsup, mean_on), (~on).astype(float)


def design(power, tsup, on, zone, i):
    sub, off = supply_features(tsup, on)
    return np.hstack([sub, off, power[:, zone == zone[i]]])


def fit(power, tsup, on, read, zone, mask=None, ridge=RIDGE):
    """-> model dict. Rows with a missing reading are skipped; `mask` limits the rows used."""
    n = len(power)
    mask = np.ones(n, bool) if mask is None else mask
    coef, icpt = [], []
    for i in range(len(zone)):
        X = design(power, tsup, on, zone, i)
        ok = mask & ~np.isnan(read[:, i])
        Xo, y = X[ok], read[ok, i]
        mu, ym = Xo.mean(0), y.mean()
        Xc = Xo - mu
        k = X.shape[1]
        A = np.vstack([Xc, np.sqrt(ridge) * np.eye(k) * np.r_[np.full(4, 0.2), np.full(4, 0.05), np.ones(k - 8)]])
        lo = np.r_[np.zeros(4), np.full(4, -np.inf), np.zeros(k - 8)]
        b = lsq_linear(A, np.r_[y - ym, np.zeros(k)], bounds=(lo, np.full(k, np.inf))).x
        coef.append(b)
        icpt.append(float(ym - mu @ b))
    return {"version": VERSION, "zone": zone.tolist(), "coef": [c.tolist() for c in coef], "intercept": icpt, "ridge": ridge}


def predict(m, power, tsup, on):
    zone = np.array(m["zone"])
    power = np.atleast_2d(power)
    out = np.empty((len(power), len(zone)))
    for i in range(len(zone)):
        out[:, i] = design(power, np.broadcast_to(tsup, (len(power), 4)), np.broadcast_to(on, (len(power), 4)), zone, i) @ np.array(m["coef"][i]) + m["intercept"][i]
    return out


def linear(m, on):
    """For a fixed CRAC state: predicted inlet = c + B @ setpoints(running units) + A @ rack power."""
    zone = np.array(m["zone"])
    on = np.asarray(on, bool)
    run = np.flatnonzero(on)
    n = len(zone)
    A, B, c = np.zeros((n, n)), np.zeros((n, len(run))), np.zeros(n)
    for i in range(n):
        b = np.array(m["coef"][i])
        sup, off, heat = b[:4], b[4:8], b[8:]
        A[i, zone == zone[i]] = heat
        for k in range(4):
            if on[k]:
                B[i, list(run).index(k)] += sup[k]
            else:
                B[i] += sup[k] / len(run)
        c[i] = m["intercept"][i] + off[~on].sum()
    return c, B, A


def dataset(h):
    """What the sensors recorded, from a world.history dict (the API builds the same arrays from the telemetry table)."""
    return {"cpu": h["read_cpu"], "gpu": h["read_gpu"], "power": h["read_cpu"] + h["read_gpu"], "tsup": h["read_supply"], "on": h["on"], "read": h["read_inlet"]}


def calibrate(d, zone, truth=None):
    """Fit on the first fifty days, check on the last ten; one refit per CRAC with its maintenance window held out says how wrong
    the model is with a unit down; then the final fit on everything. `truth` (true inlets) is used only to report error, never to fit."""
    n = len(d["power"])
    day = np.arange(n) // 24
    train = day < TRAIN_DAYS
    m = fit(d["power"], d["tsup"], d["on"], d["read"], zone, train)
    pred = predict(m, d["power"], d["tsup"], d["on"])
    val = ~train & d["on"].all(1)
    res = (d["read"] - pred)[val]
    res = res[~np.isnan(res)]
    design_rule = np.nanmean(np.where(d["on"], d["tsup"], np.nan), 1)[:, None] + 1.5        # "supply air plus a degree and a half"
    out = {"val_mae": float(np.abs(res).mean()), "val_p99": float(np.quantile(np.abs(res), 0.99)),
           "design_rule_mae": float(np.nanmean(np.abs(d["read"] - design_rule)[val])), "design_rule_misses": float(np.nanmean((d["read"] - design_rule)[val] > 2))}
    if truth is not None:
        out["truth_val_mae"] = float(np.abs(truth - pred)[val].mean())
    outage = []
    for k in range(4):
        win = ~d["on"][:, k]
        if not win.any():
            continue
        mk = fit(d["power"], d["tsup"], d["on"], d["read"], zone, ~win)
        r = (d["read"] - predict(mk, d["power"], d["tsup"], d["on"]))[win]
        r = r[~np.isnan(r)]
        outage.append({"unit": W.CRACS[k][0], "hours": int(win.sum()), "mae": float(np.abs(r).mean()), "p99": float(np.quantile(np.abs(r), 0.99)), "worst_under": float(r.max())})
    out["outage_holdout"] = outage
    final = fit(d["power"], d["tsup"], d["on"], d["read"], zone)
    zl = np.stack([d["power"][:, zone == z].sum(1) for z in (0, 1)], 1)
    final.update({"margin": round(out["val_p99"], 2), "margin_outage": round(max([out["val_p99"]] + [o["p99"] for o in outage]), 2),
                  "envelope_kw": [float(np.quantile(zl[:, z], 0.999)) for z in (0, 1)], "setpoint_range": [float(np.nanmin(d["tsup"])), float(np.nanmax(d["tsup"]))]})
    return final, out


# ---------------------------------------------------------------- heat forecast

def heat_forecast(cpu, days=14):
    """Rack CPU heat by hour of day: the mean of the last `days` days at each hour (24 x racks)."""
    last = cpu[-days * 24:]
    return last.reshape(days, 24, -1).mean(0)


# ---------------------------------------------------------------- inventory helpers

def air_delivered(caps, on):
    w = np.where(W.CRAC_ZONE[None, :] == np.arange(2)[:, None], 1 - CROSS, CROSS)
    return (w * np.asarray(caps) * np.asarray(on, bool)).sum(1)


def air_demand(cpu, gpu, zone):
    return np.array([(cpu + GPU_AIR * gpu)[zone == z].sum() for z in (0, 1)])


def extrapolating(m, cpu, gpu, zone):
    load = np.array([(cpu + gpu)[zone == z].sum() for z in (0, 1)])
    return bool((load > 1.05 * np.array(m["envelope_kw"])).any()), load


def margin(m, extrap, outage=False):
    return (m["margin_outage"] if outage else m["margin"]) * (EXTRAPOLATION_FACTOR if extrap else 1.0)


# ---------------------------------------------------------------- placement

def place_thermal(m, inv, cpu, setpoints, jobs, rule=True, on=(1, 1, 1, 1)):
    """MILP. inv: zone, row, free, gpu (installed servers), caps. With `rule`, each aisle's IT airflow stays under RULE of what it
    receives. Returns (servers added per rack, per-job racks, solver info)."""
    zone, row, free = inv["zone"], inv["row"], inv["free"]
    n, S, R = len(zone), len(jobs), W.ROWS
    c, B, A = linear(m, on)
    run = np.flatnonzero(np.asarray(on, bool))
    base = c + B @ np.asarray(setpoints, float)[run] + A @ (cpu + inv["gpu"] * W.GPU_KW)
    nv = n + S * R + 1                          # n_j, y_sr, t
    obj = np.zeros(nv)
    obj[-1] = 1.0
    obj[:n] = 0.02 / n * A.sum(0) * W.GPU_KW     # tie-break: a cooler room overall
    rows_A, lo, hi = [], [], []
    for s in range(S):
        a = np.zeros(nv)
        a[n + s * R: n + (s + 1) * R] = 1
        rows_A.append(a), lo.append(1), hi.append(1)
    for r in range(R):
        a = np.zeros(nv)
        a[:n][row == r] = 1
        for s, (_, size) in enumerate(jobs):
            a[n + s * R + r] = -size
        rows_A.append(a), lo.append(0), hi.append(0)
    for i in range(n):
        a = np.zeros(nv)
        a[:n] = A[i] * W.GPU_KW
        a[-1] = -1
        rows_A.append(a), lo.append(-np.inf), hi.append(-base[i])
    if rule:
        room = RULE * air_delivered(inv["caps"], on) - air_demand(cpu, inv["gpu"] * W.GPU_KW, zone)
        for z in (0, 1):
            a = np.zeros(nv)
            a[:n][zone == z] = GPU_AIR * W.GPU_KW
            rows_A.append(a), lo.append(-np.inf), hi.append(room[z])
    integ = np.r_[np.ones(n + S * R), 0]
    ub = np.r_[free, np.ones(S * R), np.inf]
    res = milp(obj, constraints=LinearConstraint(np.array(rows_A), lo, hi), integrality=integ, bounds=Bounds(np.zeros(nv), ub), options={"time_limit": 20})
    if res.x is None:
        return None, None, {"status": res.message}
    add = np.round(res.x[:n]).astype(int)
    y = np.round(res.x[n:n + S * R]).reshape(S, R)
    left = add.copy()
    out = []
    for s, (name, size) in enumerate(jobs):
        r = int(y[s].argmax())
        need, racks = size, {}
        for i in np.flatnonzero((row == r) & (left > 0)):
            take = min(need, left[i])
            if take:
                racks[int(i)] = int(take)
                left[i] -= take
                need -= take
            if not need:
                break
        out.append({"job": name, "row": r, "racks": racks})
    return add, out, {"status": "optimal" if res.success else res.message, "max_predicted": float(res.x[-1])}


def predicted(m, cpu, gpu_kw, setpoints, on=(1, 1, 1, 1)):
    c, B, A = linear(m, on)
    run = np.flatnonzero(np.asarray(on, bool))
    return c + B @ np.asarray(setpoints, float)[run] + A @ (cpu + gpu_kw)


# ---------------------------------------------------------------- cooling and re-planning

def cooling_plan(m, cpu, gpu_kw, on, margin, limit=W.LIMIT_C):
    """LP: the warmest running set-points that keep every predicted inlet `margin` under the limit. None if even the coldest fail."""
    c, B, A = linear(m, on)
    run = np.flatnonzero(np.asarray(on, bool))
    rhs = limit - margin - c - A @ (cpu + gpu_kw)
    res = linprog(-np.ones(len(run)), A_ub=B, b_ub=rhs, bounds=[(W.SET_MIN, W.SET_MAX)] * len(run), method="highs")
    if not res.success:
        return None
    sp = np.full(4, np.nan)
    sp[run] = np.floor(res.x * 2) / 2                 # set-points in half degrees, rounded down
    return sp


def correction(m, inv, cpu, gpu_kw, setpoints, on, read):
    """What the sensors say the model is missing after a CRAC stops. In an aisle getting more than RULE of its air, the shortfall is
    read as starvation that grows with the aisle's airflow use past RULE (kappa per rack, degrees per unit of use); elsewhere it is a
    constant offset. Returns (offset, kappa, use)."""
    off = read - predicted(m, cpu, gpu_kw, setpoints, on)
    off = np.where(np.isnan(off), 0.0, off)
    use = air_demand(cpu, gpu_kw, inv["zone"]) / air_delivered(inv["caps"], on)
    starving = use[inv["zone"]] > RULE
    kappa = np.where(starving, np.clip(off, 0, None) / np.maximum(use[inv["zone"]] - RULE, 1e-6), 0.0)
    return np.where(starving, 0.0, off), kappa, use


def replan(m, inv, cpu, gpu_kw, on, margin, rule=True, offset=None, kappa=None, limit=W.LIMIT_C):
    """LP after a CRAC stops: running set-points and a power cap per GPU rack (between CAP_MIN and 1), keeping every predicted inlet
    `margin` under the limit (and, with `rule`, each aisle's airflow under RULE of what it now receives), capping as little GPU power
    as possible and then running as warm as possible. None if no caps and set-points can do it."""
    zone = inv["zone"]
    c, B, A = linear(m, on)
    if offset is not None:
        c = c + offset
    run = np.flatnonzero(np.asarray(on, bool))
    g = np.flatnonzero(gpu_kw > 0)
    nr, ng = len(run), len(g)
    obj = np.r_[-0.01 * np.ones(nr), -gpu_kw[g]]
    A_ub = np.hstack([B, A[:, g] * gpu_kw[g]])
    b_ub = limit - margin - c - A @ cpu
    if kappa is not None and kappa.any():       # starvation shrinks as caps bring the aisle's airflow use back towards RULE
        air = air_delivered(inv["caps"], on)
        k = np.flatnonzero(kappa)
        S = np.zeros((len(k), ng))
        for r_, i in enumerate(k):
            same = zone[g] == zone[i]
            S[r_, same] = kappa[i] * GPU_AIR * gpu_kw[g][same] / air[zone[i]]
        cpu_use = np.array([cpu[zone == z].sum() for z in (0, 1)]) / air
        A_ub = np.vstack([A_ub, np.hstack([B[k], A[k][:, g] * gpu_kw[g] + S])])
        b_ub = np.r_[b_ub, b_ub[k] - kappa[k] * (cpu_use[zone[k]] - RULE)]
    if rule:
        room = RULE * air_delivered(inv["caps"], on) - air_demand(cpu, np.zeros_like(cpu), zone)
        rz = np.zeros((2, nr + ng))
        for z in (0, 1):
            rz[z, nr:] = GPU_AIR * gpu_kw[g] * (zone[g] == z)
        A_ub, b_ub = np.vstack([A_ub, rz]), np.r_[b_ub, room]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=[(W.SET_MIN, W.SET_MAX)] * nr + [(CAP_MIN, 1.0)] * ng, method="highs")
    if not res.success:
        return None
    sp = np.full(4, np.nan)
    sp[run] = np.floor(res.x[:nr] * 2) / 2
    cap = np.ones(len(zone))
    cap[g] = np.minimum(1.0, np.floor(res.x[nr:] * 20 + 1e-6) / 20)      # caps in 5% steps, rounded down
    return sp, cap


# ---------------------------------------------------------------- scoring in the simulated room

def score(rm, cpu_day, gpu_kw, setpoints, on, cap=None):
    """The room over a day: true inlets with GPUs at 95% of nameplate (or their cap), cooling and facility power."""
    gpu_true = gpu_kw * 0.95 if cap is None else np.minimum(gpu_kw * 0.95, gpu_kw * cap)
    sp = np.where(np.asarray(on, bool), setpoints, 18.0)
    t = W.inlet(rm, cpu_day, np.broadcast_to(gpu_true, cpu_day.shape), sp, on)
    it = (cpu_day + gpu_true).sum(1)
    cool = np.array([W.cooling_kw(rm, [(cpu_day[h] + gpu_true)[rm["zone"] == z].sum() for z in (0, 1)], sp, on) for h in range(len(cpu_day))])
    return {"max_inlet": float(t.max()), "racks_over_limit": int((t > W.LIMIT_C).any(0).sum()), "racks_over_allowable": int((t > W.ALLOWABLE_C).any(0).sum()),
            "rack_hours_over_limit": int((t > W.LIMIT_C).sum()), "it_kwh": float(it.sum()), "cooling_kwh": float(cool.sum()),
            "pue": float((it.sum() * (1 + W.OTHER_OVERHEAD) + cool.sum()) / it.sum()), "inlet": t, "inlet_max_by_rack": t.max(0), "hourly_max": t.max(1),
            "gpu_kw_capped": float((gpu_kw * 0.95 - gpu_true).sum()), "cooling_kw": cool, "it_kw": it}
