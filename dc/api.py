"""Public API (modular monolith). Blueprint services map to: facility telemetry (the hall, its sensors, reading checks), thermal model
(the surrogate fitted to telemetry), workload profile and placement optimizer (GPU jobs into racks), hotspot predictor (inlet forecast
per rack and hour), cooling control (set-points, power caps and the actions that change them), energy market (next-day price and
carbon), and the digital twin (what-ifs on the model, and the simulated hall that stands in for the real one).

    uvicorn dc.api:app          python -m core.jobs dc.api      # the worker
"""
import datetime
import functools
import hashlib
import json
import uuid

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.rows import tuple_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine as E, world as W

READ = {"facility:read", "jobs:read"}
OPS = READ | {"telemetry:write", "models:fit", "plans:write"}
PERMISSIONS = {"viewer": READ, "operator": OPS, "facility_manager": OPS | {"facility:load", "room:events", "actions:approve", "audit:read"}}
app = create_app("datacenter-cooling-optimizer", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
ROW_NAMES = "ABCD"
RANGES = {"inlet_temp": (5, 60), "cpu_power": (0, W.RACK_CAP_KW), "gpu_power": (0, W.RACK_CAP_KW), "supply_temp": (5, 35), "unit_running": (0, 1)}


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def r1(x):
    return round(float(x), 1)


def nn(a):
    """Array to a JSON list, NaN (a stopped unit's set-point) as null."""
    return [None if np.isnan(x) else float(x) for x in a]


def facility(c, fid):
    f = c.execute("SELECT * FROM facility WHERE facility_id = %s", [fid]).fetchone()
    if not f:
        raise Problem(404, "facility_not_found")
    return f


@functools.lru_cache(maxsize=8)
def hall(seed):
    """The simulated hall and its next day: the stand-in for the real room, never shown to a model."""
    rm = W.room(seed)
    cpu_day, _ = W.day_ahead(rm, seed)
    return rm, cpu_day


def inventory(c, fid):
    racks = c.execute("""SELECT k.rack_id, k.rack_no, k.label, k.row_no, k.col_no, k.aisle, k.gpu_slots,
                                count(s.server_id) FILTER (WHERE s.kind = 'gpu') AS gpu, coalesce(avg(s.power_cap) FILTER (WHERE s.kind = 'gpu'), 1) AS cap
                           FROM rack k JOIN room r USING (room_id) LEFT JOIN server s USING (rack_id)
                          WHERE r.facility_id = %s GROUP BY k.rack_id ORDER BY k.rack_no""", [fid]).fetchall()
    units = c.execute("SELECT u.* FROM cooling_unit u JOIN room r USING (room_id) WHERE r.facility_id = %s ORDER BY unit_no", [fid]).fetchall()
    gpu = np.array([k["gpu"] for k in racks])
    return {"ids": [k["rack_id"] for k in racks], "labels": [k["label"] for k in racks], "zone": np.array([k["aisle"] for k in racks]),
            "row": np.array([k["row_no"] for k in racks]), "col": np.array([k["col_no"] for k in racks]), "gpu": gpu,
            "free": np.array([k["gpu_slots"] for k in racks]) - gpu, "cap": np.array([float(k["cap"]) for k in racks]),
            "caps": np.array([u["airflow_kw"] for u in units]), "on": np.array([u["status"] == "running" for u in units]),
            "setpoints": np.array([u["setpoint_c"] for u in units], float), "units": units, "room_id": units[0]["room_id"]}


def series(c, f, hours=None):
    """The telemetry before the facility's clock (the last `hours` of it, or all) as hourly arrays (hours x racks, hours x units)."""
    since = f["clock"] - datetime.timedelta(hours=hours) if hours else datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
    rows = c.cursor(row_factory=tuple_row).execute("""
        SELECT s.kind, coalesce(k.rack_no, u.unit_no), extract(epoch FROM t.at)::bigint, t.value
          FROM telemetry t JOIN sensor s USING (sensor_id) LEFT JOIN rack k USING (rack_id) LEFT JOIN cooling_unit u USING (unit_id)
         WHERE s.facility_id = %s AND t.at < %s AND t.at >= %s""", [f["facility_id"], f["clock"], since]).fetchall()
    if not rows:
        raise Problem(409, "no_telemetry")
    kind = np.array([r[0] for r in rows])
    idx, ts, val = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    hours = np.unique(ts)
    hi = np.searchsorted(hours, ts)
    grid = {}
    for k, width in (("inlet_temp", W.N_RACKS), ("cpu_power", W.N_RACKS), ("gpu_power", W.N_RACKS), ("supply_temp", 4), ("unit_running", 4)):
        a = np.full((len(hours), width), np.nan)
        m = kind == k
        a[hi[m], idx[m]] = val[m]
        grid[k] = a
    on = grid["unit_running"] > 0.5
    ok = ~np.isnan(grid["cpu_power"]).any(1) & ~np.isnan(grid["gpu_power"]).any(1) & ~np.isnan(grid["unit_running"]).any(1) & ~np.isnan(np.where(on, grid["supply_temp"], 0)).any(1)
    g = {k: v[ok] for k, v in grid.items()}
    return {"cpu": g["cpu_power"], "gpu": g["gpu_power"], "power": g["cpu_power"] + g["gpu_power"], "tsup": g["supply_temp"], "on": g["unit_running"] > 0.5,
            "read": g["inlet_temp"], "hours": hours[ok]}


def current_model(c, fid):
    m = c.execute("SELECT * FROM model_artifact WHERE facility_id = %s AND approved ORDER BY created_at DESC LIMIT 1", [fid]).fetchone()
    if not m:
        raise Problem(409, "no_model", "fit and promote a thermal model first: POST /v1/thermal/models/fit")
    return m


def plan_load(c, f):
    """CPU heat at the forecast peak hour (the rack heat forecast) and the whole forecast day."""
    fc = E.heat_forecast(series(c, f, 14 * 24)["cpu"])
    return fc, int(fc.sum(1).argmax())


def prices(c, fid):
    p = c.execute("SELECT usd_per_mwh, kg_co2_per_mwh FROM energy_price WHERE facility_id = %s ORDER BY at", [fid]).fetchall()
    return np.array([x["usd_per_mwh"] for x in p]), np.array([x["kg_co2_per_mwh"] for x in p])


def energy(inv, cpu_hours, gpu_kw, setpoints, on, usd, co2):
    """The planner's estimate for a day: chiller curve from the datasheet, airflow split from commissioning."""
    plant = {"crac_cap": inv["caps"], "cross": E.CROSS}
    it = cpu_hours + gpu_kw
    cool = np.array([W.cooling_kw(plant, [h[inv["zone"] == z].sum() for z in (0, 1)], np.nan_to_num(setpoints, nan=18.0), on) for h in it])
    total = it.sum(1)
    return {"it_kwh": round(float(total.sum())), "cooling_kwh": round(float(cool.sum())), "pue": round(float((total.sum() * (1 + W.OTHER_OVERHEAD) + cool.sum()) / total.sum()), 3),
            "cooling_usd": round(float((cool * usd).sum() / 1000)), "cooling_kg_co2": round(float((cool * co2).sum() / 1000))}


def room_day(f, gpu_kw, setpoints, on, cap=None):
    rm, cpu_day = hall(f["seed"])
    return E.score(rm, cpu_day, gpu_kw, np.nan_to_num(setpoints, nan=18.0), on, cap)


def room_summary(s, inv, usd=None, co2=None):
    out = {"max_inlet": r1(s["max_inlet"]), "racks_over_27": s["racks_over_limit"], "racks_over_32": s["racks_over_allowable"], "rack_hours_over_27": s["rack_hours_over_limit"],
           "hottest_rack": inv["labels"][int(s["inlet_max_by_rack"].argmax())], "pue": round(s["pue"], 3), "cooling_kwh": round(s["cooling_kwh"])}
    if usd is not None:
        out["cooling_usd"] = round(float((s["cooling_kw"] * usd).sum() / 1000))
    if co2 is not None:
        out["cooling_kg_co2"] = round(float((s["cooling_kw"] * co2).sum() / 1000))
    return out


def decision(c, ctx, dtype, subject, model_id, inputs, result, confidence, policy):
    did = uuid.uuid7()
    digest = hashlib.sha256(json.dumps(jsonable_encoder(inputs), sort_keys=True).encode()).hexdigest()
    c.execute("""INSERT INTO decision_record (decision_id, tenant_id, subject_type, subject_id, decision_type, model_id, inputs_hash, result, confidence, policy_state, created_by)
                 VALUES (%s,%s,'facility',%s,%s,%s,%s,%s,%s,%s,%s)""", [did, ctx.tenant_id, subject, dtype, model_id, digest, Jsonb(jsonable_encoder(result)), confidence, Jsonb(policy), ctx.actor_id])
    return did


def thermal_states(c, ctx, did, inv, pred, measured):
    db.load(c, "thermal_state", ["decision_id", "rack_id", "predicted_c", "measured_c", "tenant_id"],
            [(did, inv["ids"][i], float(pred[i]), float(measured[i]), ctx.tenant_id) for i in range(len(pred))])


def rack_map(inv, pred=None, measured=None, new=None, cap=None):
    return [{"label": inv["labels"][i], "row": int(inv["row"][i]), "col": int(inv["col"][i]), "gpu": int(inv["gpu"][i]), **({"new": int(new[i])} if new is not None else {}),
             **({"predicted": r1(pred[i])} if pred is not None else {}), **({"measured": r1(measured[i])} if measured is not None else {}),
             **({"cap": round(float(cap[i]), 2)} if cap is not None else {})} for i in range(len(inv["labels"]))]


# ---------------------------------------------------------------- facility telemetry

class LoadIn(BaseModel):
    external_ref: str = Field(min_length=1, max_length=60)
    name: str = Field("Hall 1", max_length=120)
    seed: int = Field(19, ge=0, le=10 ** 6)


@app.post("/v1/facilities/load", status_code=202, tags=["facility"], summary="(+) Load the synthetic hall: 48 racks, 4 CRAC units, their sensors and sixty days of hourly telemetry, in a background job")
def load_facility(body: LoadIn, ctx: Ctx = Depends(auth("facility:load")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM facility WHERE external_ref = %s", [body.external_ref]).fetchone():
            raise Problem(409, "facility_exists")
        return 202, jobs.enqueue(c, ctx, "facility.load", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("facility.load")
def load_job(c, job):
    t, p = job["tenant_id"], job["payload"]
    h = W.history(p["seed"])
    rm = h["room"]
    fid = uuid.uuid5(t, f"facility-{p['external_ref']}")
    clock = datetime.datetime.now(datetime.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    c.execute("INSERT INTO facility (facility_id, tenant_id, external_ref, name, metadata, seed, clock) VALUES (%s,%s,%s,%s,%s,%s,%s)",
              [fid, t, p["external_ref"], p["name"], Jsonb({"source": "synthetic hall", "rows": W.ROWS, "racks": W.N_RACKS}), p["seed"], clock])
    rid = uuid.uuid5(fid, "room")
    c.execute("INSERT INTO room (room_id, tenant_id, facility_id, status, attributes, observed_at) VALUES (%s,%s,%s,'operating',%s,%s)",
              [rid, t, fid, Jsonb({"layout": "hot aisle / cold aisle, two cold aisles", "rack_pdu_kw": W.RACK_CAP_KW, "gpu_slots_per_rack": W.SLOTS}), clock])
    racks, servers, units, sensors = [], [], [], []
    for i in range(W.N_RACKS):
        k = uuid.uuid5(fid, f"rack-{i}")
        label = f"{ROW_NAMES[rm['row'][i]]}{rm['col'][i] + 1:02d}"
        racks.append((k, t, rid, i, label, int(rm["row"][i]), int(rm["col"][i]), int(rm["zone"][i]), W.RACK_CAP_KW, int(rm["gpu"][i] + rm["free"][i])))
        servers.append((uuid.uuid5(k, "cpu"), t, k, "cpu", round(float(rm["base"][i]) * 1.3, 1)))
        servers += [(uuid.uuid5(k, f"gpu-{s}"), t, k, "gpu", W.GPU_KW) for s in range(rm["gpu"][i])]
        sensors += [(uuid.uuid5(k, kind), t, f"{label}-{kind.split('_')[0]}", kind, fid, k, None) for kind in ("inlet_temp", "cpu_power", "gpu_power")]
    for n, (name, aisle, x) in enumerate(W.CRACS):
        u = uuid.uuid5(fid, name)
        units.append((u, t, rid, n, name, int(aisle), float(x), float(rm["crac_cap"][n]), "running", 18.0))
        sensors += [(uuid.uuid5(u, "supply_temp"), t, f"{name}-supply", "supply_temp", fid, None, u), (uuid.uuid5(u, "unit_running"), t, f"{name}-status", "unit_running", fid, None, u)]
    db.load(c, "rack", ["rack_id", "tenant_id", "room_id", "rack_no", "label", "row_no", "col_no", "aisle", "power_cap_kw", "gpu_slots"], racks)
    db.load(c, "server", ["server_id", "tenant_id", "rack_id", "kind", "rated_kw"], servers)
    db.load(c, "cooling_unit", ["unit_id", "tenant_id", "room_id", "unit_no", "name", "aisle", "position_x", "airflow_kw", "status", "setpoint_c"], units)
    db.load(c, "sensor", ["sensor_id", "tenant_id", "external_ref", "kind", "facility_id", "rack_id", "unit_id"], sensors)
    T = h["hours"]
    at = [clock - datetime.timedelta(hours=T - i) for i in range(T)]
    sid = {s[2]: s[0] for s in sensors}
    rows = []
    for i in range(W.N_RACKS):
        label = racks[i][4]
        for kind, arr in (("inlet", h["read_inlet"]), ("cpu", h["read_cpu"]), ("gpu", h["read_gpu"])):
            s = sid[f"{label}-{kind}"]
            rows += [(s, at[j], round(float(v), 3), t) for j, v in enumerate(arr[:, i]) if not np.isnan(v)]
    for n, (name, _, _) in enumerate(W.CRACS):
        rows += [(sid[f"{name}-supply"], at[j], round(float(v), 3), t) for j, v in enumerate(h["read_supply"][:, n]) if not np.isnan(v)]
        rows += [(sid[f"{name}-status"], at[j], float(v), t) for j, v in enumerate(h["on"][:, n])]
    db.load(c, "telemetry", ["sensor_id", "at", "value", "tenant_id"], rows)
    usd, co2 = W.prices(p["seed"])
    db.load(c, "energy_price", ["facility_id", "at", "usd_per_mwh", "kg_co2_per_mwh", "tenant_id"],
            [(fid, clock + datetime.timedelta(hours=k), float(usd[k]), float(co2[k]), t) for k in range(24)])
    audit.record(c, worker_ctx(job), "facility.loaded", "facility", fid, {"racks": W.N_RACKS, "readings": len(rows), "seed": p["seed"]})
    return {"facility_id": fid, "racks": W.N_RACKS, "cooling_units": len(units), "readings": len(rows), "days": W.HISTORY_DAYS}


@app.get("/v1/facilities/{facility_id}", tags=["facility"], summary="(+) The hall: racks with their load, GPU servers and free slots; cooling units; the last day's temperatures")
def get_facility(facility_id: uuid.UUID, ctx: Ctx = Depends(auth("facility:read"))):
    with db.tx(ctx.tenant_id) as c:
        f = facility(c, facility_id)
        inv = inventory(c, facility_id)
        d = series(c, f, 24)
        n_hours = c.execute("SELECT count(DISTINCT at) AS n FROM telemetry t JOIN sensor s USING (sensor_id) WHERE s.facility_id = %s AND t.at < %s", [facility_id, f["clock"]]).fetchone()["n"]
    day = slice(-24, None)
    cpu, gpu, read = d["cpu"][day].mean(0), d["gpu"][day].mean(0), np.nanmax(d["read"][day], 0)
    it = d["power"][day].sum(1)
    return jsonable_encoder({"facility_id": facility_id, "name": f["name"], "clock": f["clock"], "racks": len(inv["labels"]), "gpu_servers": int(inv["gpu"].sum()), "free_gpu_slots": int(inv["free"].sum()),
                             "it_kw_last_day": {"mean": r1(it.mean()), "max": r1(it.max())}, "inlet_last_day_max": r1(np.nanmax(read)), "hours_of_telemetry": n_hours,
                             "units": [{"name": u["name"], "aisle": u["aisle"] + 1, "airflow_kw": u["airflow_kw"], "status": u["status"], "setpoint_c": u["setpoint_c"]} for u in inv["units"]],
                             "aisle_airflow_use": [round(float(x), 2) for x in E.air_demand(cpu, gpu, inv["zone"]) / E.air_delivered(inv["caps"], inv["on"])],
                             "rack_map": [{**r, "cpu_kw": r1(cpu[i]), "gpu_kw": r1(gpu[i]), "free": int(inv["free"][i]), "inlet_max": r1(read[i])} for i, r in enumerate(rack_map(inv))]})


class Reading(BaseModel):
    sensor: str = Field(min_length=1, max_length=40)
    at: datetime.datetime
    value: float


class TelemetryIn(BaseModel):
    facility_id: uuid.UUID
    readings: list[Reading] = Field(min_length=1, max_length=5000)


@app.post("/v1/telemetry", status_code=201, tags=["facility"], summary="Sensor readings, each checked on its own: unknown sensor, no time zone, from the future, out of range, duplicate")
def telemetry(body: TelemetryIn, ctx: Ctx = Depends(auth("telemetry:write")), idem: str | None = IdemKey):
    def work(c):
        facility(c, body.facility_id)
        sensors = {s["external_ref"]: s for s in c.execute("SELECT sensor_id, external_ref, kind FROM sensor WHERE facility_id = %s", [body.facility_id])}
        now = datetime.datetime.now(datetime.UTC)
        accepted, rejected, seen = [], [], set()
        for r in body.readings:
            s = sensors.get(r.sensor)
            lo, hi = RANGES[s["kind"]] if s else (0, 0)
            why = ("unknown sensor" if not s else "timestamp has no time zone" if r.at.tzinfo is None else "timestamp is in the future" if r.at > now + datetime.timedelta(minutes=5)
                   else f"{r.value} is outside {lo}–{hi} for {s['kind']}" if not lo <= r.value <= hi
                   else "duplicate reading" if (r.sensor, r.at) in seen or c.execute("SELECT 1 FROM telemetry WHERE sensor_id = %s AND at = %s", [s["sensor_id"], r.at]).fetchone() else None)
            if why:
                rejected.append({"sensor": r.sensor, "at": r.at, "why": why})
                continue
            seen.add((r.sensor, r.at))
            accepted.append((s["sensor_id"], r.at, r.value, ctx.tenant_id))
        if accepted:
            db.load(c, "telemetry", ["sensor_id", "at", "value", "tenant_id"], accepted)
        audit.record(c, ctx, "telemetry.ingested", "facility", body.facility_id, {"accepted": len(accepted), "rejected": len(rejected)})
        return 201, {"accepted": len(accepted), "rejected": rejected}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- thermal model

class FitIn(BaseModel):
    facility_id: uuid.UUID


@app.post("/v1/thermal/models/fit", status_code=201, tags=["thermal model"], summary="(+) Fit the thermal surrogate to the telemetry, check it on the last ten days and against each CRAC's maintenance window, and promote it if it passes")
def fit_model(body: FitIn, ctx: Ctx = Depends(auth("models:fit")), idem: str | None = IdemKey):
    def work(c):
        f = facility(c, body.facility_id)
        inv = inventory(c, body.facility_id)
        d = series(c, f)
        m, ev = E.calibrate(d, inv["zone"])
        use = np.stack([E.air_demand(d["cpu"][t], d["gpu"][t], inv["zone"]) / E.air_delivered(inv["caps"], d["on"][t]) for t in range(len(d["cpu"]))])
        m["outage_use_seen"] = float(use[~d["on"].all(1)].max()) if (~d["on"].all(1)).any() else 0.0
        accept = ev["val_p99"] < 1.5 and ev["val_mae"] < ev["design_rule_mae"]
        ver = f"{E.VERSION}-{int(datetime.datetime.now().timestamp() * 1000)}"
        mid = uuid.uuid7()
        c.execute("""INSERT INTO model_artifact (model_id, tenant_id, model_name, version, artifact_uri, feature_schema, metrics, approved, facility_id, params)
                     VALUES (%s,%s,'thermal_surrogate',%s,%s,%s,%s,%s,%s,%s)""",
                  [mid, ctx.tenant_id, ver, f"db://model_artifact/thermal_surrogate/{ver}", Jsonb({"per_rack": ["supply air of each CRAC (a stopped unit takes the running units' mean)", "CRAC stopped flags", "power of every rack in the same aisle"]}),
                   Jsonb(ev), accept, body.facility_id, Jsonb(m)])
        audit.record(c, ctx, "model.fitted", "model_artifact", ver, {"val_mae": round(ev["val_mae"], 3), "promoted": accept})
        # evidence: one rack's last three days, and how far each rack's inlet rises if its whole aisle draws 10 kW more per rack (model and simulation truth)
        rm, _ = hall(f["seed"])
        hot = int(np.nanargmax(np.nanmean(d["read"][-240:], 0)))
        pred = E.predict(m, d["power"], d["tsup"], d["on"])[-72:, hot]
        rule = np.nanmean(np.where(d["on"], d["tsup"], np.nan), 1)[-72:] + 1.5
        _, _, A = E.linear(m, np.ones(4, bool))
        truth_A = W.recirculation(rm)
        h = W.history(f["seed"])
        truth_mae = float(np.abs(h["inlet_true"][-240:] - E.predict(m, d["power"], d["tsup"], d["on"])[-240:]).mean()) if len(h["inlet_true"]) == len(d["read"]) else None
        return 201, {"version": ver, "model_id": mid, "promoted": accept, "hours": len(d["hours"]), "trained_on_days": E.TRAIN_DAYS, "checked_on_days": W.HISTORY_DAYS - E.TRAIN_DAYS,
                     "validation": {"mae": round(ev["val_mae"], 3), "p99": round(ev["val_p99"], 2), "design_rule_mae": round(ev["design_rule_mae"], 3), "design_rule_off_by_2c": round(ev["design_rule_misses"], 3)},
                     "outage_holdout": [{k: round(v, 2) if isinstance(v, float) else v for k, v in o.items()} for o in ev["outage_holdout"]],
                     "margin_c": m["margin"], "margin_outage_c": m["margin_outage"], "envelope_kw": [round(x) for x in m["envelope_kw"]], "setpoint_range": m["setpoint_range"],
                     "outage_airflow_use_seen": round(m["outage_use_seen"], 2),
                     "acceptance": "p99 error under 1.5 °C and better than the design rule (supply air + 1.5 °C)",
                     "trace": {"rack": inv["labels"][hot], "read": [None if np.isnan(x) else r1(x) for x in d["read"][-72:, hot]], "model": [r1(x) for x in pred], "design_rule": [r1(x) for x in rule]},
                     "recirculation": [{"label": inv["labels"][i], "row": int(inv["row"][i]), "col": int(inv["col"][i]), "model": round(float(A[i].sum() * 10), 2), "truth": round(float(truth_A[i].sum() * 10), 2)} for i in range(len(inv["labels"]))],
                     "simulation_truth": {"what": "the true inlet temperatures behind the readings, which the fit never sees", "mae_last_ten_days": round(truth_mae, 3) if truth_mae is not None else None}}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- placement and hotspots

class Job(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    servers: int = Field(ge=1, le=16)
    priority: int = Field(1, ge=0, le=9)


class PlaceIn(BaseModel):
    facility_id: uuid.UUID
    jobs: list[Job] = Field(min_length=1, max_length=20)


@app.post("/v1/placements/optimize", status_code=201, tags=["placement"], summary="Place GPU training jobs (each within one row for its network) two ways, the scheduler's first fit and thermal-aware, predict the hotspots of each and adopt the thermal-aware plan")
def optimize(body: PlaceIn, ctx: Ctx = Depends(auth("plans:write")), idem: str | None = IdemKey):
    def work(c):
        f = facility(c, body.facility_id)
        mrow = current_model(c, body.facility_id)
        m = mrow["params"]
        inv = inventory(c, body.facility_id)
        jobs_ = [(j.name, j.servers) for j in body.jobs]
        if sum(j.servers for j in body.jobs) > inv["free"].sum():
            raise Problem(409, "no_room", f"{sum(j.servers for j in body.jobs)} servers asked for, {int(inv['free'].sum())} free slots")
        fc, ph = plan_load(c, f)
        cpu = fc[ph]
        sp, on = inv["setpoints"], inv["on"]
        rm, _ = hall(f["seed"])
        ff_add, ff_jobs = W.first_fit({"free": inv["free"], "row": inv["row"]}, jobs_)
        if any(j["row"] is None for j in ff_jobs):
            raise Problem(409, "no_row_fits", "a job is larger than any row's free slots")
        ta_add, ta_jobs, info = E.place_thermal(m, inv, cpu, sp, jobs_, on=on)
        if ta_add is None:
            raise Problem(409, "no_safe_placement", info["status"])
        wl = {}
        for j in body.jobs:
            w = uuid.uuid7()
            c.execute("INSERT INTO workload (workload_id, tenant_id, facility_id, name, servers, kw_per_server, priority, status) VALUES (%s,%s,%s,%s,%s,%s,%s,'queued')",
                      [w, ctx.tenant_id, body.facility_id, j.name, j.servers, W.GPU_KW, j.priority])
            wl[j.name] = w
        out = {}
        for strategy, add, placed in (("first_fit", ff_add, ff_jobs), ("thermal_aware", ta_add, ta_jobs)):
            g = (inv["gpu"] + add) * W.GPU_KW
            pred = E.predicted(m, cpu, g, sp, on)
            xp, load = E.extrapolating(m, cpu, g, inv["zone"])
            mg = E.margin(m, xp)
            room = room_day(f, g, sp, on)
            use = E.air_demand(cpu, g, inv["zone"]) / E.air_delivered(inv["caps"], on)
            result = {"strategy": strategy, "gpu_kw": g.tolist(), "plan_hour": ph, "setpoints": sp.tolist()}
            did = decision(c, ctx, "placement", body.facility_id, mrow["model_id"], {"jobs": jobs_, "strategy": strategy, "cpu": cpu.round(2).tolist()}, result,
                           0.5 if xp else 0.9, {"extrapolating": xp, "margin_c": mg, "airflow_rule": E.RULE if strategy == "thermal_aware" else None})
            thermal_states(c, ctx, did, inv, pred, room["inlet"][ph])
            db.load(c, "placement", ["decision_id", "workload_id", "rack_id", "servers", "tenant_id"],
                    [(did, wl[p["job"]], inv["ids"][i], n, ctx.tenant_id) for p in placed for i, n in p["racks"].items()])
            out[strategy] = {"decision_id": did, "jobs": [{"job": p["job"], "row": ROW_NAMES[p["row"]], "racks": [{"rack": inv["labels"][i], "servers": n} for i, n in p["racks"].items()]} for p in placed],
                             "predicted_max_inlet": r1(pred.max()), "hotspots": [inv["labels"][i] for i in np.flatnonzero(pred + mg > W.LIMIT_C)], "margin_c": round(mg, 2), "extrapolating": xp,
                             "aisle_kw": [round(float(x)) for x in load], "aisle_airflow_use": [round(float(u), 2) for u in use],
                             "racks": rack_map(inv, pred, room["inlet"][ph], add),
                             "room": {**room_summary(room, inv), "what": "the simulated hall over the next day at today's set-points; no model sees it"}}
        # adopt the thermal-aware plan: the servers are racked and the jobs placed
        adopted = out["thermal_aware"]
        servers = [(uuid.uuid7(), ctx.tenant_id, inv["ids"][i], "gpu", W.GPU_KW, wl[p["job"]]) for p in ta_jobs for i, n in p["racks"].items() for _ in range(n)]
        db.load(c, "server", ["server_id", "tenant_id", "rack_id", "kind", "rated_kw", "workload_id"], servers)
        c.execute("UPDATE workload SET status = 'placed' WHERE workload_id = ANY(%s)", [list(wl.values())])
        audit.record(c, ctx, "placement.decided", "decision_record", adopted["decision_id"], {"jobs": len(jobs_), "servers": len(servers), "predicted_max_inlet": adopted["predicted_max_inlet"]})
        return 201, {"adopted": "thermal_aware", "plan_hour": ph, "setpoints": sp.tolist(), "rule": E.RULE, "limit_c": W.LIMIT_C, "servers": len(servers), "gpu_kw_added": len(servers) * W.GPU_KW,
                     "solver": info, **out,
                     "how": "the thermal surrogate predicts every rack's inlet at the forecast peak hour; first fit is the scheduler's default (first row with room, racks filled from the row's west end); thermal-aware is a MILP over servers per rack and the row of each job that minimises the hottest predicted inlet, with each aisle's airflow kept under 85% of what its units deliver because the model is outside the load it was fitted on"}
    return run(ctx, idem, body, work)


@app.get("/v1/thermal/forecast", tags=["hotspots"], summary="Hourly inlet forecast for the next day under a placement: the hottest predicted rack each hour with its margin, the hotspots, and what the simulated hall did")
def forecast(decision_id: uuid.UUID, ctx: Ctx = Depends(auth("facility:read"))):
    with db.tx(ctx.tenant_id) as c:
        dr = c.execute("SELECT * FROM decision_record WHERE decision_id = %s", [decision_id]).fetchone()
        if not dr:
            raise Problem(404, "decision_not_found")
        f = facility(c, dr["subject_id"])
        inv = inventory(c, f["facility_id"])
        m = c.execute("SELECT params FROM model_artifact WHERE model_id = %s", [dr["model_id"]]).fetchone()["params"]
        fc, ph = plan_load(c, f)
    r = dr["result"]
    g, sp = np.array(r["gpu_kw"]), np.array(r["setpoints"], float)
    on = np.ones(4, bool)
    pred = np.stack([E.predicted(m, fc[h], g, sp, on) for h in range(24)])
    mg = dr["policy_state"]["margin_c"]
    rm, cpu_day = hall(f["seed"])
    room = room_day(f, g, sp, on)
    hot = sorted({inv["labels"][i] for i in np.flatnonzero((pred + mg > W.LIMIT_C).any(0))})
    return jsonable_encoder({"decision_id": decision_id, "strategy": r["strategy"], "margin_c": mg, "limit_c": W.LIMIT_C,
                             "hours": [{"hour": h, "predicted_max": r1(pred[h].max()), "measured_max": r1(room["hourly_max"][h]), "it_kw": r1(fc[h].sum() + g.sum())} for h in range(24)],
                             "hotspots": hot, "peak_hour": ph,
                             "heat_forecast": {"how": "each rack's CPU heat at each hour: the mean of the last fourteen days at that hour", "mae_kw": round(float(np.abs(fc - cpu_day).mean()), 3), "rack_hours": int(fc.size)},
                             "simulation_truth": {"what": "the simulated hall's next day: true CPU heat, GPUs at 95% of nameplate", "racks_over_27": room["racks_over_limit"], "max_inlet": r1(room["max_inlet"])}})


# ---------------------------------------------------------------- cooling control

class CoolIn(BaseModel):
    facility_id: uuid.UUID


@app.post("/v1/cooling/plan", status_code=201, tags=["cooling"], summary="Raise the CRAC set-points as far as the predicted inlets allow, with the safety margin; apply them (bounded automation) and price the day against fixed 18 °C")
def cooling(body: CoolIn, ctx: Ctx = Depends(auth("plans:write")), idem: str | None = IdemKey):
    def work(c):
        f = facility(c, body.facility_id)
        mrow = current_model(c, body.facility_id)
        m = mrow["params"]
        inv = inventory(c, body.facility_id)
        fc, ph = plan_load(c, f)
        g = inv["gpu"] * W.GPU_KW * inv["cap"]
        on = inv["on"]
        xp, _ = E.extrapolating(m, fc[ph], g, inv["zone"])
        mg = E.margin(m, xp)
        sp = E.cooling_plan(m, fc[ph], g, on, mg)
        if sp is None:
            raise Problem(409, "no_safe_setpoints", "even the coldest set-points leave a predicted inlet over the limit")
        usd, co2 = prices(c, body.facility_id)
        pred = E.predicted(m, fc[ph], g, sp, on)
        before = inv["setpoints"]
        cold = np.where(on, 18.0, np.nan)
        room, room18 = room_day(f, g, sp, on), room_day(f, g, cold, on)
        did = decision(c, ctx, "cooling_plan", body.facility_id, mrow["model_id"], {"setpoints": nn(before), "gpu_kw": g.tolist()},
                       {"setpoints": nn(sp), "plan_hour": ph, "gpu_kw": g.tolist()}, 0.5 if xp else 0.9, {"extrapolating": xp, "margin_c": mg, "mode": "automatic", "bounds": [W.SET_MIN, W.SET_MAX]})
        thermal_states(c, ctx, did, inv, pred, room["inlet"][ph])
        actions = []
        for u, s in zip(inv["units"], sp):
            if u["status"] == "running" and s != u["setpoint_c"]:
                c.execute("""INSERT INTO control_action (action_id, tenant_id, decision_id, kind, target, value, previous, mode, status, applied_at)
                             VALUES (%s,%s,%s,'setpoint',%s,%s,%s,'automatic','applied',now())""", [uuid.uuid7(), ctx.tenant_id, did, u["name"], float(s), u["setpoint_c"]])
                c.execute("UPDATE cooling_unit SET setpoint_c = %s WHERE unit_id = %s", [float(s), u["unit_id"]])
                actions.append({"unit": u["name"], "from": u["setpoint_c"], "to": float(s)})
        est, est18 = energy(inv, fc, g, sp, on, usd, co2), energy(inv, fc, g, cold, on, usd, co2)
        audit.record(c, ctx, "cooling.planned", "decision_record", did, {"setpoints": nn(sp), "pue_estimate": est["pue"]})
        units = [{"name": u["name"], "aisle": u["aisle"] + 1, "status": u["status"], "setpoint_c": None if np.isnan(x) else float(x)} for u, x in zip(inv["units"], sp)]
        return 201, {"decision_id": did, "actions": actions, "units": units, "mode": "automatic: every predicted inlet stays the margin under 27 °C, set-points within 16–27 °C",
                     "predicted_max_inlet": r1(pred.max()), "margin_c": round(mg, 2), "extrapolating": xp, "estimate": est, "estimate_at_18c": est18,
                     "price": {"usd_per_mwh": [round(float(x), 1) for x in usd], "kg_co2_per_mwh": [round(float(x)) for x in co2]},
                     "room": room_summary(room, inv, usd, co2), "room_at_18c": room_summary(room18, inv, usd, co2), "racks": rack_map(inv, pred, room["inlet"][ph]),
                     "how": "an LP over the running units' set-points maximising their sum, each rack's predicted inlet at the forecast peak hour at most 27 °C less the margin; energy from the chiller's COP curve and the commissioning airflow split"}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- digital twin and the simulated hall

class SimIn(BaseModel):
    facility_id: uuid.UUID
    fail_unit: str = Field("CRAC-1", pattern="^CRAC-[1-4]$")


@app.post("/v1/twin/simulate", tags=["digital twin"], summary="What-if on the model: one CRAC stops; predicted inlets with no action and with the runbook (all running units to 16 °C), and how far outside its experience the model is")
def simulate(body: SimIn, ctx: Ctx = Depends(auth("plans:write"))):
    with db.tx(ctx.tenant_id) as c:
        f = facility(c, body.facility_id)
        m = current_model(c, body.facility_id)["params"]
        inv = inventory(c, body.facility_id)
        fc, ph = plan_load(c, f)
    k = [u["name"] for u in inv["units"]].index(body.fail_unit)
    on = inv["on"].copy()
    on[k] = False
    g = inv["gpu"] * W.GPU_KW * inv["cap"]
    use = E.air_demand(fc[ph], g, inv["zone"]) / E.air_delivered(inv["caps"], on)
    out = {}
    for name, sp in (("no_action", inv["setpoints"]), ("runbook_16c", np.full(4, W.SET_MIN))):
        pred = E.predicted(m, fc[ph], g, sp, on)
        room = room_day(f, g, sp, on)
        out[name] = {"predicted_max_inlet": r1(pred.max()), "predicted_over_27": int((pred > W.LIMIT_C).sum()), "simulation_truth": room_summary(room, inv)}
    aisle = int(W.CRAC_ZONE[k])
    outside = bool(use[aisle] > 1.05 * m["outage_use_seen"])
    return jsonable_encoder({"fail_unit": body.fail_unit, "aisle": aisle + 1, "aisle_airflow_use": round(float(use[aisle]), 2), "outage_airflow_use_seen": round(m["outage_use_seen"], 2),
                             "confidence": "low" if outside else "normal", **out,
                             "recommendation": "the model has never seen an aisle this short of air: do not act on its prediction; apply the runbook, read the sensors, re-plan from what they show" if outside else "the model has seen this; plan from its prediction"})


class EventIn(BaseModel):
    facility_id: uuid.UUID
    kind: str = Field("crac_failure", pattern="^crac_failure$")
    unit: str = Field(pattern="^CRAC-[1-4]$")


@app.post("/v1/room/events", status_code=202, tags=["digital twin"], summary="(+) Simulated hall only: a CRAC stops. The automatic response (all running units to 16 °C) is applied, the hall's sensors report, and a re-plan from the readings proposes set-points and GPU power caps for approval")
def room_event(body: EventIn, ctx: Ctx = Depends(auth("room:events")), idem: str | None = IdemKey):
    def work(c):
        facility(c, body.facility_id)
        u = c.execute("SELECT u.status FROM cooling_unit u JOIN room r USING (room_id) WHERE r.facility_id = %s AND u.name = %s", [body.facility_id, body.unit]).fetchone()
        if not u or u["status"] != "running":
            raise Problem(409, "unit_not_running")
        return 202, jobs.enqueue(c, ctx, "room.event", body.model_dump(mode="json"))
    return run(ctx, idem, body, work)


@jobs.handler("room.event")
def event_job(c, job):
    p = job["payload"]
    ctx = worker_ctx(job)
    fid = uuid.UUID(p["facility_id"])
    f = facility(c, fid)
    mrow = current_model(c, fid)
    m = mrow["params"]
    inv = inventory(c, fid)
    fc, ph = plan_load(c, f)
    k = [u["name"] for u in inv["units"]].index(p["unit"])
    before = inv["setpoints"].copy()
    c.execute("UPDATE cooling_unit SET status = 'failed' WHERE unit_id = %s", [inv["units"][k]["unit_id"]])
    on = inv["on"].copy()
    on[k] = False
    g = inv["gpu"] * W.GPU_KW * inv["cap"]
    open_loop = E.predicted(m, fc[ph], g, before, on)
    # 1. the automatic response: every running unit to its coldest set-point
    rb = np.where(on, W.SET_MIN, np.nan)
    did = decision(c, ctx, "failure_response", fid, mrow["model_id"], {"unit": p["unit"], "setpoints": nn(before)}, {"unit": p["unit"]}, None, {})
    for n, u in enumerate(inv["units"]):
        if on[n]:
            c.execute("""INSERT INTO control_action (action_id, tenant_id, decision_id, kind, target, value, previous, mode, status, applied_at)
                         VALUES (%s,%s,%s,'setpoint',%s,%s,%s,'automatic','applied',now())""", [uuid.uuid7(), ctx.tenant_id, did, u["name"], W.SET_MIN, u["setpoint_c"]])
            c.execute("UPDATE cooling_unit SET setpoint_c = %s WHERE unit_id = %s", [W.SET_MIN, u["unit_id"]])
    # 2. the hall's sensors report ten minutes later (at the day's forecast peak hour)
    rm, cpu_day = hall(f["seed"])
    r = np.random.default_rng(f["seed"] + 3)
    gpu_now = g * 0.95
    true_now = W.inlet(rm, cpu_day[ph], gpu_now, np.nan_to_num(rb, nan=18.0), on)[0]
    read = true_now + rm["bias"] + r.normal(0, 0.25, W.N_RACKS)
    cpu_read, gpu_read = cpu_day[ph] * (1 + 0.01 * r.standard_normal(W.N_RACKS)), gpu_now * (1 + 0.01 * r.standard_normal(W.N_RACKS))
    at = f["clock"] + datetime.timedelta(hours=ph, minutes=10)
    sens = {s["external_ref"]: s["sensor_id"] for s in c.execute("SELECT sensor_id, external_ref FROM sensor WHERE facility_id = %s", [fid])}
    rows = [(sens[f"{lab}-{kind}"], at, round(float(v[i]), 3), ctx.tenant_id) for i, lab in enumerate(inv["labels"]) for kind, v in (("inlet", read), ("cpu", cpu_read), ("gpu", gpu_read))]
    rows += [(sens[f"{u['name']}-status"], at, float(on[n]), ctx.tenant_id) for n, u in enumerate(inv["units"])]
    c.execute("DELETE FROM telemetry WHERE at = %s AND sensor_id = ANY(%s)", [at, [x[0] for x in rows]])       # a re-run of the job replaces its readings
    db.load(c, "telemetry", ["sensor_id", "at", "value", "tenant_id"], rows)
    # 3. re-plan from the readings
    offset, kappa, use = E.correction(m, inv, cpu_read, gpu_read, rb, on, read)
    plan = E.replan(m, inv, fc[ph], g, on, m["margin_outage"], rule=False, offset=offset, kappa=kappa)
    feasible = plan is not None
    if feasible:
        sp, cap = plan
    else:       # review: the most the automation may do, and a person decides what to pause
        sp, cap = rb, np.where((g > 0) & (use[inv["zone"]] > E.RULE), E.CAP_MIN, 1.0)
    needs_caps = bool((cap < 1).any())
    mode = "needs_approval" if needs_caps else "automatic"
    actions = []
    for n, u in enumerate(inv["units"]):
        if on[n] and sp[n] != W.SET_MIN:
            actions.append(("setpoint", u["name"], float(sp[n]), W.SET_MIN))
    for i in np.flatnonzero(cap < 1):
        actions.append(("power_cap", inv["labels"][i], float(cap[i]), 1.0))
    for kind, target, value, prev in actions:
        c.execute("""INSERT INTO control_action (action_id, tenant_id, decision_id, kind, target, value, previous, mode, status, applied_at)
                     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", [uuid.uuid7(), ctx.tenant_id, did, kind, target, value, prev, mode,
                                                                 "applied" if mode == "automatic" else "pending_approval", datetime.datetime.now(datetime.UTC) if mode == "automatic" else None])
        if mode == "automatic" and kind == "setpoint":
            c.execute("UPDATE cooling_unit SET setpoint_c = %s WHERE room_id = %s AND name = %s", [value, inv["room_id"], target])
    after = E.predicted(m, fc[ph], g * cap, sp, on) + offset + kappa * np.clip(E.air_demand(fc[ph], g * cap, inv["zone"]) / E.air_delivered(inv["caps"], on) - E.RULE, 0, None)[inv["zone"]]
    room_now = room_day(f, g, rb, on)
    thermal_states(c, ctx, did, inv, after, read)
    result = {"unit": p["unit"], "plan_hour": ph, "setpoints": nn(sp), "cap": cap.tolist(), "gpu_kw": g.tolist(), "feasible": feasible}
    c.execute("UPDATE decision_record SET result = %s, confidence = %s, policy_state = %s WHERE decision_id = %s",
              [Jsonb(result), 0.8 if feasible else 0.2, Jsonb({"mode": mode, "margin_c": m["margin_outage"], "review": not feasible}), did])
    audit.record(c, ctx, "room.crac_failed", "cooling_unit", p["unit"], {"automatic": "running units to 16 °C", "max_inlet_read": r1(np.nanmax(read))})
    audit.record(c, ctx, "failure.replanned", "decision_record", did, {"feasible": feasible, "power_caps": int((cap < 1).sum()), "mode": mode})
    return {"decision_id": did, "unit": p["unit"], "aisle": int(W.CRAC_ZONE[k]) + 1, "model_said": {"max_inlet": r1(open_loop.max()), "over_27": int((open_loop > W.LIMIT_C).sum())},
            "automatic": {"setpoints": nn(rb), "read_at": at, "max_inlet_read": r1(np.nanmax(read)), "racks_read_over_27": int((read > W.LIMIT_C).sum()),
                          "aisle_airflow_use": [round(float(x), 2) for x in use], "rack_map": rack_map(inv, measured=read)},
            "replan": {"feasible": feasible, "mode": mode, "setpoints": nn(sp), "power_caps": [{"rack": inv["labels"][i], "cap": float(cap[i]), "gpu_kw": r1(g[i]), "kw_capped": r1(g[i] * (1 - cap[i]))} for i in np.flatnonzero(cap < 1)],
                       "gpu_kw_capped": r1((g * (1 - cap)).sum()), "predicted_max_inlet": r1(after.max()), "margin_c": m["margin_outage"], "rack_map": rack_map(inv, after, cap=cap)},
            "room_until_approved": {**room_summary(room_now, inv), "what": "the simulated hall over the day if it stays on the automatic response"}}


class ApproveIn(BaseModel):
    note: str = Field(min_length=3, max_length=500)


@app.post("/v1/decisions/{decision_id}/approve", tags=["cooling"], summary="(+) Approve a decision's pending control actions (GPU power caps and the set-points that assume them): the facility manager only")
def approve(decision_id: uuid.UUID, body: ApproveIn, ctx: Ctx = Depends(auth("plans:write")), idem: str | None = IdemKey):
    def work(c):
        dr = c.execute("SELECT * FROM decision_record WHERE decision_id = %s", [decision_id]).fetchone()
        if not dr:
            raise Problem(404, "decision_not_found")
        if "actions:approve" not in PERMISSIONS[ctx.role]:
            raise Problem(403, "forbidden", "power caps slow customers' training jobs: approval needs the facility manager")
        pending = c.execute("SELECT * FROM control_action WHERE decision_id = %s AND status = 'pending_approval' ORDER BY kind, target", [decision_id]).fetchall()
        if not pending:
            raise Problem(409, "nothing_pending")
        f = facility(c, dr["subject_id"])
        inv = inventory(c, f["facility_id"])
        for a in pending:
            if a["kind"] == "setpoint":
                c.execute("UPDATE cooling_unit SET setpoint_c = %s WHERE room_id = %s AND name = %s", [a["value"], inv["room_id"], a["target"]])
            else:
                c.execute("UPDATE server SET power_cap = %s WHERE kind = 'gpu' AND rack_id = %s", [a["value"], inv["ids"][inv["labels"].index(a["target"])]])
        c.execute("UPDATE control_action SET status = 'applied', approved_by = %s, applied_at = now() WHERE decision_id = %s AND status = 'pending_approval'", [ctx.actor_id, decision_id])
        r = dr["result"]
        inv = inventory(c, f["facility_id"])
        g = np.array(r["gpu_kw"])
        room = room_day(f, g, inv["setpoints"], inv["on"], np.array(r["cap"]))
        c.execute("UPDATE thermal_state SET measured_c = x.m FROM unnest(%s::uuid[], %s::real[]) AS x(id, m) WHERE decision_id = %s AND rack_id = x.id",
                  [inv["ids"], [float(v) for v in room["inlet"][r["plan_hour"]]], decision_id])
        audit.record(c, ctx, "decision.approved", "decision_record", decision_id, {"note": body.note, "actions": len(pending), "max_inlet_after": r1(room["max_inlet"])})
        return 200, {"decision_id": decision_id, "approved_by": ctx.actor_name, "applied": [{"kind": a["kind"], "target": a["target"], "value": a["value"]} for a in pending],
                     "room_after": {**room_summary(room, inv), "gpu_kw_capped": r1(room["gpu_kw_capped"]), "what": "the simulated hall over the day with the caps and set-points applied"}}
    return run(ctx, idem, body, work)


@app.get("/v1/facilities/{facility_id}/pue", tags=["energy"], summary="Power usage effectiveness: the last week as metered, and the next day as planned now, with cooling cost and carbon at the next-day prices")
def pue(facility_id: uuid.UUID, ctx: Ctx = Depends(auth("facility:read"))):
    with db.tx(ctx.tenant_id) as c:
        f = facility(c, facility_id)
        inv = inventory(c, facility_id)
        d = series(c, f, 14 * 24)
        fc = E.heat_forecast(d["cpu"])
        usd, co2 = prices(c, facility_id)
    plant = {"crac_cap": inv["caps"], "cross": E.CROSS}
    wk = slice(-168, None)
    cool = [W.cooling_kw(plant, [d["power"][t][inv["zone"] == z].sum() for z in (0, 1)], np.nan_to_num(d["tsup"][t], nan=18.0), d["on"][t]) for t in range(len(d["power"]))[wk]]
    it = d["power"][wk].sum(1)
    g = inv["gpu"] * W.GPU_KW
    est = energy(inv, fc, g * inv["cap"], inv["setpoints"], inv["on"], usd, co2)
    room = room_day(f, g, inv["setpoints"], inv["on"], inv["cap"])
    return jsonable_encoder({"facility_id": facility_id, "last_week": {"pue": round(float((it.sum() * (1 + W.OTHER_OVERHEAD) + sum(cool)) / it.sum()), 3), "it_kw_mean": r1(it.mean()),
                                                                     "how": "metered IT power, cooling from the supply temperatures and the chiller curve, 6% for UPS losses and lighting"},
                             "now": {"units": [{"name": u["name"], "status": u["status"], "setpoint_c": u["setpoint_c"]} for u in inv["units"]], "gpu_servers": int(inv["gpu"].sum()),
                                     "gpu_capped_racks": int((inv["cap"] < 1).sum()), "estimate": est,
                                     "simulation_truth": room_summary(room, inv, usd)},
                             "prices": {"usd_per_mwh_mean": r1(usd.mean()), "kg_co2_per_mwh_mean": r1(co2.mean())}})
