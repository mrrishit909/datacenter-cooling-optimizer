"""Integration, end-to-end and security tests against a real Postgres."""
import psycopg
import pytest

from core import db, jobs, scenario

from .conftest import bearer

MGR, OPS, VIEWER, OTHER = bearer("hall", "facility_manager"), bearer("hall", "operator"), bearer("hall", "viewer"), bearer("other", "facility_manager")


def test_nothing_runs_before_a_hall_and_a_model(client):
    assert client.post("/v1/thermal/models/fit", headers=OPS, json={"facility_id": "00000000-0000-0000-0000-000000000000"}).json()["code"] == "facility_not_found"
    assert client.post("/v1/facilities/load", headers=OPS, json={"external_ref": "X"}).status_code == 403


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    assert demo["load"]["result"]["readings"] > 200000 and demo["hall"]["gpu_servers"] == 27
    assert demo["tele"]["accepted"] == 2 and [r["why"].split()[0] for r in demo["tele"]["rejected"]] == ["unknown", "85.0", "timestamp"]
    fit = demo["fit"]
    assert fit["promoted"] and fit["validation"]["mae"] < fit["validation"]["design_rule_mae"] and len(fit["outage_holdout"]) == 4
    p = demo["place"]
    assert p["servers"] == 24 and one("SELECT count(*) AS n FROM server WHERE kind = 'gpu'")["n"] == 27 + 24
    assert p["thermal_aware"]["predicted_max_inlet"] < p["first_fit"]["predicted_max_inlet"] and p["thermal_aware"]["extrapolating"]
    assert {j["row"] for j in p["first_fit"]["jobs"]} == {"A"} and one("SELECT count(*) AS n FROM thermal_state JOIN decision_record USING (decision_id) WHERE decision_type = 'placement'")["n"] == 2 * 48
    assert demo["fc"]["simulation_truth"]["racks_over_27"] == 0 and len(demo["fc"]["hours"]) == 24
    cool = demo["cool"]
    assert cool["actions"] and cool["estimate"]["cooling_kwh"] < cool["estimate_at_18c"]["cooling_kwh"] and cool["room"]["racks_over_27"] == 0
    w = demo["whatif"]
    assert w["confidence"] == "low" and w["no_action"]["predicted_over_27"] == 0 and w["no_action"]["simulation_truth"]["racks_over_27"] > 0
    f = demo["fail"]["result"]
    assert f["automatic"]["racks_read_over_27"] > 0 and f["replan"]["feasible"] and f["replan"]["mode"] == "needs_approval" and f["replan"]["power_caps"]
    assert f["room_until_approved"]["racks_over_27"] > 0
    assert demo["self"]["code"] == "forbidden" and demo["approve"]["room_after"]["racks_over_27"] == 0
    assert one("SELECT count(*) AS n FROM control_action WHERE status <> 'applied'")["n"] == 0 and one("SELECT count(*) AS n FROM server WHERE power_cap < 1")["n"] > 0
    assert demo["pue"]["now"]["units"][0]["status"] == "failed"
    assert demo["audit"]["chain_valid"] and {"facility.loaded", "telemetry.ingested", "model.fitted", "placement.decided", "cooling.planned", "room.crac_failed", "failure.replanned", "decision.approved"} <= {e["action"] for e in demo["audit"]["events"]}


def test_validation_replay_and_approval_rules(client, demo):
    fid = demo["load"]["result"]["facility_id"]
    did = demo["fail"]["result"]["decision_id"]
    assert client.post(f"/v1/decisions/{did}/approve", headers=MGR, json={"note": "again"}).json()["code"] == "nothing_pending"
    assert client.post("/v1/room/events", headers=MGR, json={"facility_id": fid, "unit": "CRAC-1"}).json()["code"] == "unit_not_running"
    assert client.post("/v1/placements/optimize", headers=OPS, json={"facility_id": fid, "jobs": [{"name": "huge", "servers": 16}] * 12}).json()["code"] == "no_room"
    r = client.post("/v1/telemetry", headers=OPS, json={"facility_id": fid, "readings": [{"sensor": "A01-inlet", "at": "2026-01-01T00:00:00", "value": 20}]}).json()
    assert r["rejected"][0]["why"] == "timestamp has no time zone"
    h = {**OPS, "Idempotency-Key": "plan-replay"}
    a, b = client.post("/v1/cooling/plan", headers=h, json={"facility_id": fid}), client.post("/v1/cooling/plan", headers=h, json={"facility_id": fid})
    assert a.status_code == 201 and a.json() == b.json() and b.headers.get("Idempotent-Replay") == "true"
    assert client.post("/v1/cooling/plan", headers=h, json={"facility_id": "00000000-0000-0000-0000-000000000000"}).json()["code"] == "idempotency_key_reused"


def test_tenant_isolation_and_append_only_audit(client, demo):
    fid = demo["load"]["result"]["facility_id"]
    assert client.get(f"/v1/facilities/{fid}", headers=OTHER).status_code == 404
    assert client.get(f"/v1/thermal/forecast?decision_id={demo['place']['thermal_aware']['decision_id']}", headers=OTHER).status_code == 404
    with db.tx() as c:
        other = c.execute("SELECT DISTINCT tenant_id FROM api_tokens WHERE actor_name LIKE '%%Tamsin%%'").fetchone()["tenant_id"]
    assert one("SELECT count(*) AS n FROM telemetry", tenant=other)["n"] == 0 and one("SELECT count(*) AS n FROM rack", tenant=other)["n"] == 0
    with pytest.raises(psycopg.errors.RaiseException), db.tx() as c:
        c.execute("UPDATE audit_events SET action = 'edited'")
