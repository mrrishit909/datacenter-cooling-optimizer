import os

import psycopg
import pytest

ADMIN = os.environ.get("DATABASE_URL", "postgresql://app:app-dev-only@127.0.0.1:55410/app")
os.environ["DATABASE_URL"] = ADMIN.rsplit("/", 1)[0] + "/app_test"


@pytest.fixture(scope="session")
def seeded():
    """A fresh database with two tenants and their tokens. The hall arrives through the load job in the tests."""
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS app_test WITH (FORCE)")
        c.execute("CREATE DATABASE app_test")
    from core import db
    from dc import seed
    out = seed.main()
    yield out
    db.close()


@pytest.fixture(scope="session")
def client(seeded):
    from fastapi.testclient import TestClient

    from dc.api import app
    return TestClient(app)


def bearer(tenant, role):
    return {"Authorization": f"Bearer {tenant}-{role}-demo"}
