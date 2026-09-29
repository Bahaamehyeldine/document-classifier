"""Auth, roles, invitations and the audit trail."""

import pytest
from sqlalchemy import create_engine, text

from tests.api.conftest import TEST_DATABASE_URL, make_user, register


async def test_health(client):
    assert (await client.get("/health/live")).json() == {"status": "ok"}
    assert (await client.get("/health/ready")).status_code == 200


async def test_register_and_login_without_role_has_no_permissions(client):
    headers, _ = await register(client, "nobody@example.com")
    me = (await client.get("/me", headers=headers)).json()
    assert me["roles"] == [] and me["permissions"] == []
    assert (await client.get("/batches", headers=headers)).status_code == 403


async def test_unauthenticated_requests_are_rejected(client):
    for path in ("/me", "/batches", "/predictions/recent", "/audit", "/users"):
        assert (await client.get(path)).status_code == 401, path


@pytest.mark.parametrize(
    "role, path, allowed",
    [
        ("admin", "/users", True),
        ("admin", "/audit", True),
        ("admin", "/batches", True),
        ("reviewer", "/users", False),
        ("reviewer", "/audit", False),
        ("reviewer", "/batches", True),
        ("auditor", "/users", False),
        ("auditor", "/audit", True),
        ("auditor", "/batches", True),
    ],
)
async def test_permission_matrix(client, role, path, allowed):
    headers, _ = await make_user(client, f"{role}@example.com", role)
    status = (await client.get(path, headers=headers)).status_code
    assert (status == 200) is allowed, (role, path, status)


async def test_invitation_grants_role_on_registration_and_is_audited(client):
    admin, _ = await make_user(client, "admin@example.com", "admin")
    r = await client.post(
        "/users/invitations", json={"email": "Rev@Example.com", "role": "reviewer"}, headers=admin
    )
    assert r.status_code == 201 and r.json()["accepted_at"] is None

    reviewer, _ = await register(client, "rev@example.com")
    assert (await client.get("/me", headers=reviewer)).json()["roles"] == ["reviewer"]

    actions = [e["action"] for e in (await client.get("/audit", headers=admin)).json()]
    assert actions[:2] == ["role.changed", "invitation.created"]


async def test_role_toggle_applies_on_next_request_and_invalidates_me_cache(client):
    admin, _ = await make_user(client, "admin@example.com", "admin")
    user, user_id = await make_user(client, "ann@example.com", "auditor")

    first = await client.get("/me", headers=user)
    assert first.json()["roles"] == ["auditor"]
    cached = await client.get("/me", headers=user)
    assert cached.headers.get("X-FastAPI-Cache") == "HIT"

    r = await client.put(f"/users/{user_id}/role", json={"role": "reviewer"}, headers=admin)
    assert r.status_code == 200 and r.json() == ["reviewer"]

    after = await client.get("/me", headers=user)  # same token, no re-login
    assert after.json()["roles"] == ["reviewer"]
    assert (await client.get("/audit", headers=user)).status_code == 403  # auditor right is gone

    entry = (await client.get("/audit?action=role.changed", headers=admin)).json()[0]
    assert entry["actor"] == "admin@example.com"
    assert entry["details"]["from"] == ["auditor"] and entry["details"]["to"] == ["reviewer"]
    assert entry["request_id"] == r.headers["X-Request-ID"]


async def test_last_admin_cannot_be_demoted(client):
    admin, admin_id = await make_user(client, "admin@example.com", "admin")
    r = await client.put(f"/users/{admin_id}/role", json={"role": None}, headers=admin)
    assert r.status_code == 409


async def test_only_admins_change_roles(client):
    reviewer, rid = await make_user(client, "rev@example.com", "reviewer")
    r = await client.put(f"/users/{rid}/role", json={"role": "admin"}, headers=reviewer)
    assert r.status_code == 403


async def test_request_id_is_propagated(client):
    r = await client.get("/health/live", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


async def test_audit_log_has_no_write_endpoints(client):
    admin, _ = await make_user(client, "admin@example.com", "admin")
    for method in ("post", "put", "delete"):
        assert (await getattr(client, method)("/audit", headers=admin)).status_code == 405


def test_policy_is_seeded_by_migration(migrated_db):
    eng = create_engine(TEST_DATABASE_URL)
    with eng.connect() as c:
        n = c.execute(text("SELECT count(*) FROM casbin_rule WHERE ptype='p'")).scalar()
    eng.dispose()
    assert n == 11
