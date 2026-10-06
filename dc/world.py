"""The simulated machine room: the hidden truth every model is scored against and none is fitted to.

A room of 4 rows x 12 racks in two cold aisles (rows 0-1 face aisle A, rows 2-3 face aisle B), four CRAC units at the ends of the
aisles. A rack's inlet air is a mix of the CRACs' supply air (more from the nearer unit, a little from the other aisle's), plus hot
exhaust that recirculates over and around the racks (most at the row ends), plus, when an aisle is asked for more air than its CRACs
deliver, hot air drawn back into the racks furthest from a working unit. This is not CFD: it is a lumped model with the same three
mechanisms, written so that the room can be generated, measured with noisy sensors and then used to judge what the models predict.

Sensors read inlet temperature at the top of each rack with a fixed calibration offset per sensor and noise; rack PDUs read power
with 1% noise; CRACs report their supply temperature. That is all the models ever see.
"""
import numpy as np

ROWS, COLS = 4, 12
N_RACKS = ROWS * COLS
SLOTS = 4                       # GPU server slots per rack
GPU_KW = 10.0                   # an 8-GPU training server at full load (nameplate)
GPU_IDLE_KW = 1.5
RACK_CAP_KW = 44.0              # PDU limit per rack
LIMIT_C = 27.0                  # ASHRAE recommended inlet maximum
ALLOWABLE_C = 32.0              # ASHRAE A1 allowable maximum
SET_MIN, SET_MAX = 16.0, 27.0   # CRAC supply set-point range
DELTA_T = 12.0                  # typical exhaust minus supply, degrees C
CRACS = [("CRAC-1", 0, -1.5), ("CRAC-2", 0, 12.5), ("CRAC-3", 1, -1.5), ("CRAC-4", 1, 12.5)]     # name, aisle, x (rack widths)
CRAC_ZONE = np.array([c[1] for c in CRACS])
CRAC_X = np.array([c[2] for c in CRACS])
HISTORY_DAYS = 60
FAN_KW = 7.5                    # per running CRAC
OTHER_OVERHEAD = 0.06           # UPS losses and lighting, share of IT power


def cop(t_supply):
    """Coefficient of performance of chilled-water cooling against supply temperature (the HP Labs CRAC curve)."""
    return 0.0068 * t_supply ** 2 + 0.0008 * t_supply + 0.458


def room(seed):
    r = np.random.default_rng(seed)
    row, col = np.divmod(np.arange(N_RACKS), COLS)
    base = r.uniform(2.5, 7.0, N_RACKS)                                       # CPU and storage kW
    gpu = np.where(r.random(N_RACKS) < 0.4, r.integers(1, 3, N_RACKS), 0)    # GPU servers already installed
    fit = np.floor((RACK_CAP_KW - base) / GPU_KW).astype(int)                 # how many 10 kW servers the PDU allows
    free = np.clip(np.minimum(SLOTS, fit) - gpu, 0, None)
    return {"seed": seed, "row": row, "col": col, "zone": row // 2, "x": col.astype(float), "base": base, "gpu": gpu, "free": free,
            "crac_cap": np.round(r.uniform(235, 265, len(CRACS))),          # airflow, in kW of IT it can carry (commissioning figure)
            # the hidden physics
            "rho": r.uniform(0.028, 0.040), "end": r.uniform(0.6, 1.0), "L": r.uniform(5.0, 7.0), "cross": 0.1, "knee": r.uniform(0.86, 0.94),
            "gamma": r.uniform(1.3, 1.9), "gain": 0.4, "bias": r.normal(0, 0.4, N_RACKS)}


def recirculation(rm):
    """D[i, j]: degrees C at rack i's inlet per kW of rack j, within an aisle; strongest at the row ends."""
    x, z, row = rm["x"], rm["zone"], rm["row"]
    edge = np.where((rm["col"] == 0) | (rm["col"] == COLS - 1), rm["end"], np.where((rm["col"] == 1) | (rm["col"] == COLS - 2), 0.3, 0.0))
    D = rm["rho"] * np.exp(-np.abs(x[:, None] - x[None, :]) / 1.5) * (1 + edge[:, None])
    D *= np.where(row[:, None] == row[None, :], 1.0, 0.5) * (z[:, None] == z[None, :])
    return D


def supply_share(rm, on):
    raw = np.exp(-np.abs(rm["x"][:, None] - CRAC_X[None, :]) / rm["L"]) * np.where(rm["zone"][:, None] == CRAC_ZONE[None, :], 1.0, rm["cross"]) * on
    return raw / raw.sum(1, keepdims=True)


def airflow(rm, on):
    """Air each aisle receives, in kW-of-IT: 90% of a CRAC's air goes to its own aisle, 10% to the other."""
    w = np.where(CRAC_ZONE[None, :] == np.arange(2)[:, None], 1 - rm["cross"], rm["cross"])
    return (w * rm["crac_cap"] * on).sum(1)


def starved(rm, on):
    """Which racks draw hot air back when their aisle is short of air: those furthest from a running unit (mean 1 per aisle)."""
    near = (np.exp(-np.abs(rm["x"][:, None] - CRAC_X[None, :]) / rm["L"]) * (rm["zone"][:, None] == CRAC_ZONE[None, :]) * on).max(1)
    w = 1.0 - near
    for z in (0, 1):
        w[rm["zone"] == z] /= w[rm["zone"] == z].mean()
    return w


def inlet(rm, cpu_kw, gpu_kw, setpoints, on):
    """True inlet temperatures. cpu_kw, gpu_kw: (T, racks) or (racks,); setpoints, on: (T, 4) or (4,)."""
    cpu_kw, gpu_kw = np.atleast_2d(cpu_kw), np.atleast_2d(gpu_kw)
    setpoints, on = np.atleast_2d(setpoints).astype(float), np.atleast_2d(on).astype(bool)
    setpoints = np.broadcast_to(setpoints, (len(cpu_kw), 4))
    on = np.broadcast_to(on, (len(cpu_kw), 4))
    D = recirculation(rm)
    out = (cpu_kw + gpu_kw) @ D.T + rm["gain"]
    for state in {tuple(o) for o in on}:
        m = (on == state).all(1)
        o = np.array(state, float)
        out[m] += setpoints[m] @ supply_share(rm, o).T
        demand = np.stack([(cpu_kw[m] + 0.8 * gpu_kw[m])[:, rm["zone"] == z].sum(1) for z in (0, 1)], 1)
        deficit = np.clip(demand / airflow(rm, o) - rm["knee"], 0, None)
        out[m] += rm["gamma"] * DELTA_T * deficit[:, rm["zone"]] * starved(rm, o)
    return out


def cooling_kw(rm, it_kw_by_zone, setpoints, on):
    """Compressor plus fan power: each aisle's heat is removed by the running units in proportion to the air they send it."""
    on = np.asarray(on, bool)
    w = np.where(CRAC_ZONE[None, :] == np.arange(2)[:, None], 1 - rm["cross"], rm["cross"]) * rm["crac_cap"] * on
    heat = (np.asarray(it_kw_by_zone)[:, None] * w / w.sum(1, keepdims=True)).sum(0)
    return float((heat[on] / cop(np.asarray(setpoints, float)[on])).sum() + FAN_KW * on.sum())


def history(seed, days=HISTORY_DAYS):
    """Hourly operation for `days`: diurnal CPU load, GPU servers switching between training and idle, daily set-point changes
    by the operators, and one eight-hour maintenance outage per CRAC. Returns the truth and what the sensors read."""
    rm = room(seed)
    r = np.random.default_rng(seed + 1)
    T = days * 24
    hour = np.arange(T) % 24
    cpu = rm["base"] * (1 + 0.1 * np.sin(2 * np.pi * (hour[:, None] - 9) / 24)) * (1 + 0.04 * r.standard_normal((T, N_RACKS)))
    busy = np.zeros((T, N_RACKS))
    for i in np.flatnonzero(rm["gpu"]):
        for s in range(rm["gpu"][i]):
            state, t = r.random() < 0.6, 0
            while t < T:
                run = int(r.exponential(36 if state else 18)) + 1
                busy[t:t + run, i] += 0.95 if state else GPU_IDLE_KW / GPU_KW
                t, state = t + run, not state
    gpu = busy * GPU_KW
    day = np.arange(T) // 24
    daily = np.clip(18 + np.round(r.normal(0, 1.3, (days, 4)) * 2) / 2, SET_MIN, 22)
    sp = daily[day]
    on = np.ones((T, 4), bool)
    for k in range(4):
        start = (8 + 14 * k) * 24 + 9
        on[start:start + 8, k] = False
    true_in = inlet(rm, cpu, gpu, sp, on)
    read_in = true_in + rm["bias"] + r.normal(0, 0.25, true_in.shape)
    read_in[r.random(read_in.shape) < 0.005] = np.nan
    supply = np.where(on, sp + r.normal(0, 0.15, sp.shape), np.nan)
    return {"room": rm, "hours": T, "cpu": cpu, "gpu": gpu, "setpoint": sp, "on": on, "inlet_true": true_in, "read_inlet": read_in,
            "read_cpu": cpu * (1 + 0.01 * r.standard_normal(cpu.shape)), "read_gpu": gpu * (1 + 0.01 * r.standard_normal(gpu.shape)), "read_supply": supply}


SURGE = [("llm-pretrain", 8), ("vision-ft", 4), ("rec-sys", 4), ("speech", 4), ("rl-sweep", 2), ("eval", 2)]


def day_ahead(rm, seed, hours=24):
    """The next day as the room really runs it: CPU on its diurnal curve with noise, every GPU server training at 95% of nameplate."""
    r = np.random.default_rng(seed + 7)
    h = np.arange(hours)
    cpu = rm["base"] * (1 + 0.1 * np.sin(2 * np.pi * (h[:, None] - 9) / 24)) * (1 + 0.04 * r.standard_normal((hours, N_RACKS)))
    return cpu, rm["gpu"] * GPU_KW * 0.95


def prices(seed, hours=24):
    """Next-day wholesale price ($/MWh) and grid carbon intensity (kg CO2/MWh): an evening price peak, a midday solar dip in carbon."""
    r = np.random.default_rng(seed + 11)
    h = np.arange(hours)
    usd = 55 + 45 * np.exp(-((h - 18) / 2.5) ** 2) + 8 * np.sin(2 * np.pi * (h - 8) / 24) + r.normal(0, 3, hours)
    co2 = 430 - 160 * np.clip(np.sin(np.pi * (h - 6) / 12), 0, None) + r.normal(0, 10, hours)
    return usd, co2


def first_fit(rm, jobs=SURGE):
    """The scheduler's default: each job to the first row with room for all of it, racks filled in order from the row's west end."""
    free, add = rm["free"].copy(), np.zeros(N_RACKS, int)
    out = []
    for name, n in jobs:
        row = next((r for r in range(ROWS) if free[rm["row"] == r].sum() >= n), None)
        if row is None:
            out.append({"job": name, "row": None, "racks": {}})
            continue
        left, racks = n, {}
        for i in np.flatnonzero(rm["row"] == row):
            take = min(left, free[i])
            if take:
                racks[int(i)] = int(take)
                free[i] -= take
                add[i] += take
                left -= take
        out.append({"job": name, "row": row, "racks": racks})
    return add, out
