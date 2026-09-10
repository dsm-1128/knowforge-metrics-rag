from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app import create_app
from webapp.auth_store import AuthStore


class FakeRag:
    ready = True
    detail = "测试运行时"

    def __init__(self):
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield {"type": "end", "answer": "测试答案", "sources": []}

    def messages(self, session_id):
        return []

    def clear(self, session_id):
        pass


@pytest.fixture
def service():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store = AuthStore(engine)
    store.initialize()
    for name, tenant, role in [("alice", "alpha", "tenant_admin"), ("bob", "alpha", "viewer"), ("carol", "beta", "tenant_admin")]:
        store.create_user(name, "correct-password-123", name, tenant, role, "sales", "internal")
    rag = FakeRag()
    app = create_app(store=store, runtime=rag, allowed_origins={"http://testserver"})
    with TestClient(app) as client:
        yield client, store, rag


def login(client, username="alice"):
    result = client.post("/api/auth/login", headers={"Origin": "http://testserver"}, json={"username": username, "password": "correct-password-123"})
    assert result.status_code == 200, result.text
    return {"Origin": "http://testserver", "X-CSRF-Token": result.json()["csrf_token"]}


def test_login_required_and_no_legacy_routes(service):
    client, _, _ = service
    for path in ["/api/categories", "/api/sessions", "/api/auth/me", "/api/admin/users", "/api/status"]:
        assert client.get(path).status_code == 401
    for path in ["/api/history/any", "/api/scenarios", "/api/retrieval/debug"]:
        assert client.get(path).status_code in (404, 405)


def test_csrf_and_origin_required(service):
    client, _, _ = service
    assert client.post("/api/auth/login", headers={"Origin": "https://evil.example"}, json={"username":"alice","password":"correct-password-123"}).status_code == 403
    headers = login(client)
    assert client.post("/api/sessions", json={}).status_code == 403
    assert client.post("/api/sessions", json={}, headers=headers).status_code == 200


def test_identity_cannot_be_supplied_by_client(service):
    client, _, _ = service
    headers = login(client, "bob")
    assert client.post("/api/sessions", headers=headers, json={"tenant_id":"beta"}).status_code == 422
    assert client.get("/api/auth/me").json()["user"]["tenant_id"] == "alpha"
    assert client.get("/api/admin/users").status_code == 403


def test_chat_history_owner_boundary(service):
    client, _, _ = service
    headers = login(client)
    session = client.post("/api/sessions", json={}, headers=headers).json()
    headers = login(client, "bob")
    assert client.get(f"/api/sessions/{session['id']}/messages").status_code == 404
    assert client.delete(f"/api/sessions/{session['id']}", headers=headers).status_code == 404


def test_admin_is_tenant_scoped(service):
    client, _, _ = service
    headers = login(client)
    users = client.get("/api/admin/users").json()["users"]
    assert {user["username"] for user in users} == {"alice","bob"}
    assert client.post("/api/admin/users", headers=headers, json={"username":"newuser","password":"new-password-123","display_name":"New","role":"viewer","dataset_id":"sales","visibility":"internal","tenant_id":"beta"}).status_code == 422
    login(client, "carol")
    carol = client.get("/api/auth/me").json()["user"]
    headers = login(client)
    assert client.patch(f"/api/admin/users/{carol['id']}",headers=headers,json={"enabled":False}).status_code == 404


def test_logout_revokes_session(service):
    client, _, _ = service
    headers = login(client)
    cookie = client.cookies.get("kf_session")
    assert client.post("/api/auth/logout",headers=headers).status_code == 200
    client.cookies.set("kf_session",cookie)
    assert client.get("/api/auth/me").status_code == 401


def test_websocket_scope_is_server_bound(service):
    client, _, rag = service
    headers = login(client, "bob")
    session = client.post("/api/sessions",headers=headers,json={}).json()
    with client.websocket_connect(f"/api/stream?csrf_token={headers['X-CSRF-Token']}",headers={"Origin":"http://testserver"}) as ws:
        ws.send_json({"session_id":session["id"],"query":"GMV是什么？","source_filter":"metrics","tenant_id":"beta"})
        assert ws.receive_json()["type"] == "error"
        assert not rag.calls
        ws.send_json({"session_id":session["id"],"query":"GMV是什么？","source_filter":"metrics"})
        assert ws.receive_json()["type"] == "end"
    assert rag.calls[0]["tenant_id"] == "alpha"
    assert rag.calls[0]["dataset_id"] == "sales"
    assert rag.calls[0]["visibility"] == "internal"
    assert rag.calls[0]["user_roles"] == ["public", "viewer"]


def test_websocket_foreign_session_not_retrieved(service):
    client, _, rag = service
    headers = login(client)
    session = client.post("/api/sessions",headers=headers,json={}).json()
    headers = login(client,"carol")
    with client.websocket_connect(f"/api/stream?csrf_token={headers['X-CSRF-Token']}",headers={"Origin":"http://testserver"}) as ws:
        ws.send_json({"session_id":session["id"],"query":"GMV是什么？"})
        assert ws.receive_json()["type"] == "error"
    assert not rag.calls


def test_websocket_rejects_missing_auth_or_wrong_csrf(service):
    from starlette.websockets import WebSocketDisconnect
    client, _, rag = service
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/stream?csrf_token=bad", headers={"Origin":"http://testserver"}):
            pass
    login(client)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/stream?csrf_token=bad", headers={"Origin":"http://testserver"}):
            pass
    assert rag.calls == []


def test_generation_claim_blocks_delete_and_deleted_session_cannot_be_claimed(service):
    from webapp.stream import claim_session, release_session, Query
    client, store, _ = service
    headers = login(client)
    user = client.get("/api/auth/me").json()["user"]
    session = client.post("/api/sessions", json={}, headers=headers).json()
    query = Query(query="指标是什么", session_id=session["id"])
    state = client.app.state
    assert claim_session(state, query, user) is None
    assert client.delete("/api/sessions/" + session["id"], headers=headers).status_code == 409
    release_session(state, session["id"])
    assert client.delete("/api/sessions/" + session["id"], headers=headers).status_code == 200
    assert claim_session(state, query, user) == "会话不存在或无权访问"
