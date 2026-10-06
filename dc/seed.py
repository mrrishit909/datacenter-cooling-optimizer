"""Tenants (one per operator) and demo tokens. The hall arrives through POST /v1/facilities/load.

    python -m dc.seed [--if-empty]
"""
import hashlib
import sys
import uuid

from core import db

TENANTS = {"hall": "Larkspur Compute (invented)", "other": "Tamsin Colocation (invented)"}
ROLES = ["viewer", "operator", "facility_manager"]


def token(tenant, role):
    """Demo credentials for local use only."""
    return f"{tenant}-{role}-demo"


def main(if_empty=False):
    db.migrate()
    out = {}
    with db.tx() as c:
        if if_empty and c.execute("SELECT 1 FROM tenants LIMIT 1").fetchone():
            return print("already seeded")
        for key, name in TENANTS.items():
            t = uuid.uuid5(uuid.NAMESPACE_DNS, f"{key}.datacenter-cooling-optimizer.example")
            c.execute("INSERT INTO tenants (id, name) VALUES (%s,%s)", [t, name])
            for role in ROLES:
                c.execute("INSERT INTO api_tokens VALUES (%s,%s,%s,%s,%s)", [hashlib.sha256(token(key, role).encode()).hexdigest(), t, uuid.uuid5(t, role), f"{role} ({name})", role])
            out[key] = t
    print("demo tokens:", ", ".join(token(t, r) for t in TENANTS for r in ROLES))
    return out


if __name__ == "__main__":
    main("--if-empty" in sys.argv)
