"""The simulated hall's physics, the surrogate, the optimizers and the closed loop, without a database."""
import numpy as np

from dc import engine as E, world as W

ON = np.ones(4, bool)


def test_hall_physics():
    rm = W.room(19)
    assert np.allclose(W.supply_share(rm, ON).sum(1), 1) and (W.recirculation(rm) >= 0).all()
    D = W.recirculation(rm)
    assert D[0].sum() > D[5].sum()                                     # the row end recirculates more than mid-row
    cpu, gpu = rm["base"], rm["gpu"] * W.GPU_KW
    t18, t20 = W.inlet(rm, cpu, gpu, np.full(4, 18.0), ON)[0], W.inlet(rm, cpu, gpu, np.full(4, 20.0), ON)[0]
    assert np.allclose(t20 - t18, 2.0)                                 # supply air shifts every inlet one for one
    off = ON.copy()
    off[0] = False
    heavy = gpu + 40 * (rm["zone"] == 0)
    assert W.inlet(rm, cpu, heavy, np.full(4, 18.0), off)[0].max() > W.inlet(rm, cpu, heavy, np.full(4, 18.0), ON)[0].max() + 3     # a starved aisle
    assert W.cooling_kw(rm, [300, 300], np.full(4, 24.0), ON) < W.cooling_kw(rm, [300, 300], np.full(4, 18.0), ON)                   # warmer air, better COP


def test_surrogate_learns_from_readings_and_beats_the_rule():
    h = W.history(21)
    m, ev = E.calibrate(E.dataset(h), h["room"]["zone"], h["inlet_true"])
    assert ev["val_mae"] < 0.3 and ev["val_mae"] < ev["design_rule_mae"] / 2 and ev["truth_val_mae"] < 0.5
    assert len(ev["outage_holdout"]) == 4 and m["margin_outage"] >= m["margin"] > 0
    assert all((np.array(c[:4]) >= 0).all() and (np.array(c[8:]) >= 0).all() for c in m["coef"])     # no negative heat or supply terms


def setup(seed=21):
    h = W.history(seed)
    rm = h["room"]
    d = E.dataset(h)
    m, _ = E.calibrate(d, rm["zone"])
    fc = E.heat_forecast(d["cpu"])
    inv = {"zone": rm["zone"], "row": rm["row"], "free": rm["free"], "gpu": rm["gpu"], "caps": rm["crac_cap"]}
    return rm, m, inv, fc[fc.sum(1).argmax()], int(fc.sum(1).argmax())


def test_placement_respects_slots_rows_and_airflow():
    rm, m, inv, cpu, _ = setup()
    add, jobs, info = E.place_thermal(m, inv, cpu, np.full(4, 18.0), W.SURGE)
    assert info["status"] == "optimal" and add.sum() == sum(n for _, n in W.SURGE) and (add <= inv["free"]).all()
    for (name, n), j in zip(W.SURGE, jobs):
        assert sum(j["racks"].values()) == n and {int(rm["row"][i]) for i in j["racks"]} == {j["row"]}
    g = (rm["gpu"] + add) * W.GPU_KW
    assert (E.air_demand(cpu, g, rm["zone"]) <= E.RULE * E.air_delivered(rm["crac_cap"], ON) + 1e-6).all()
    ff, _ = W.first_fit(rm)
    pred = lambda a: E.predicted(m, cpu, (rm["gpu"] + a) * W.GPU_KW, np.full(4, 18.0)).max()     # noqa: E731
    assert pred(add) < pred(ff)


def test_cooling_plan_keeps_the_margin():
    rm, m, inv, cpu, _ = setup()
    g = rm["gpu"] * W.GPU_KW
    sp = E.cooling_plan(m, cpu, g, ON, 1.0)
    assert ((sp >= W.SET_MIN) & (sp <= W.SET_MAX)).all() and E.predicted(m, cpu, g, sp).max() <= W.LIMIT_C - 1.0 + 1e-6
    assert E.cooling_plan(m, cpu, g + 200, ON, 1.0) is None or E.cooling_plan(m, cpu, g + 200, ON, 1.0).mean() < sp.mean()


def test_closed_loop_reads_corrects_and_caps_only_what_it_must():
    rm, m, inv, cpu, ph = setup(19)
    add, _, _ = E.place_thermal(m, inv, cpu, np.full(4, 18.0), W.SURGE)
    g = (rm["gpu"] + add) * W.GPU_KW
    cpu_day, _ = W.day_ahead(rm, 19)
    on = ON.copy()
    on[0] = False
    rb = np.where(on, W.SET_MIN, np.nan)
    read = W.inlet(rm, cpu_day[ph], g * 0.95, np.full(4, W.SET_MIN), on)[0] + rm["bias"]
    assert read.max() > W.LIMIT_C and E.predicted(m, cpu, g, rb, on).max() < W.LIMIT_C     # the model alone misses it
    offset, kappa, use = E.correction(m, inv, cpu_day[ph], g * 0.95, rb, on, read)
    assert use[0] > E.RULE and kappa[rm["zone"] == 0].max() > 0 and not kappa[rm["zone"] == 1].any()
    sp, cap = E.replan(m, inv, cpu, g, on, m["margin_outage"], rule=False, offset=offset, kappa=kappa)
    assert ((cap >= E.CAP_MIN) & (cap <= 1)).all() and (cap[g == 0] == 1).all() and (cap < 1).any()
    after = E.score(rm, cpu_day, g, np.nan_to_num(sp, nan=18.0), on, cap)
    assert after["racks_over_limit"] == 0 and after["gpu_kw_capped"] < 0.5 * (g * 0.95).sum()
