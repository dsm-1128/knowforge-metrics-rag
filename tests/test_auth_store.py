from datetime import datetime, timedelta, timezone
import hashlib

import pytest
from sqlalchemy import create_engine, inspect, select, update

from webapp.auth_store import AuthStore


@pytest.fixture
def store():
    result = AuthStore(create_engine('sqlite://'))
    result.initialize()
    return result


def user(store, username='alice', tenant='one'):
    return store.create_user(username, 'strong-password-123', 'Alice', tenant)


def test_explicit_initialize():
    engine = create_engine('sqlite://')
    store = AuthStore(engine)
    assert inspect(engine).get_table_names() == []
    store.initialize()
    assert len(inspect(engine).get_table_names()) == 3


def test_password_storage_and_normalization(store):
    created = user(store, ' ALICE ')
    assert created['username'] == 'alice'
    assert 'password_hash' not in created
    with store.engine.connect() as conn:
        row = conn.execute(select(store.users)).mappings().one()
    assert row['password_hash'] != 'strong-password-123'
    assert row['password_hash'].startswith('scrypt$')
    user(store, 'different')
    with store.engine.connect() as conn:
        hashes = conn.execute(select(store.users.c.password_hash)).scalars().all()
    assert len(set(hashes)) == 2
    with pytest.raises(ValueError):
        user(store, 'alice')
    with pytest.raises(ValueError):
        store.create_user('bob', 'short', 'Bob', 'one')
    with pytest.raises(ValueError):
        store.create_user('bob', 'x' * 257, 'Bob', 'one')
    with pytest.raises(ValueError):
        store.create_user('bob', 'long-password-123', 'Bob', 'one', role='root')


def test_login_token_logout_and_invalid_credentials(store):
    created = user(store)
    for name, password in [('nobody', 'strong-password-123'), ('alice', 'wrong')]:
        with pytest.raises(ValueError, match='Invalid credentials'):
            store.login(name, password)
    token, csrf, account = store.login('ALICE', 'strong-password-123')
    assert account == created
    assert store.authenticate(token)['csrf_token'] == csrf
    with store.engine.connect() as conn:
        row = conn.execute(select(store.tokens)).mappings().one()
    assert row['token_hash'] == hashlib.sha256(token.encode()).hexdigest()
    assert token not in row.values()
    store.logout(token)
    assert store.authenticate(token) is None


def test_disable_and_expiry(store):
    created = user(store)
    token, _, _ = store.login('alice', 'strong-password-123')
    with store.engine.begin() as conn:
        conn.execute(update(store.tokens).values(expires_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)))
    assert store.authenticate(token) is None
    token, _, _ = store.login('alice', 'strong-password-123')
    assert not store.set_enabled('other', created['id'], False)
    assert store.authenticate(token)
    assert store.set_enabled('one', created['id'], False)
    assert store.authenticate(token) is None
    with pytest.raises(ValueError, match='Invalid credentials'):
        store.login('alice', 'strong-password-123')
    store.set_enabled('one', created['id'], True)
    assert store.authenticate(token) is None


def test_tenant_and_session_ownership(store):
    alice = user(store)
    bob = user(store, 'bob')
    user(store, 'carol', 'two')
    assert len(store.list_users('one')) == 2
    session = store.create_session(alice['id'], 'one')
    assert store.get_session(session['id'], alice['id'], 'one')['dataset_id'] == 'default'
    assert len(store.list_sessions(alice['id'], 'one')) == 1
    assert store.list_sessions(bob['id'], 'one') == []
    assert store.get_session(session['id'], alice['id'], 'two') is None
    assert store.get_session(session['id'], bob['id'], 'one') is None
    assert not store.delete_session(session['id'], bob['id'], 'one')
    with pytest.raises(ValueError):
        store.create_session(alice['id'], 'two')
    assert store.delete_session(session['id'], alice['id'], 'one')
    assert store.list_sessions(alice['id'], 'one') == []
