"""Portable account and ownership persistence. Call initialize explicitly for DDL.

login raises ValueError('Invalid credentials') for all rejected credentials.
All timestamps are naive UTC for portable MySQL/SQLite DateTime storage.
"""
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import (Boolean, Column, DateTime, ForeignKey, Integer, MetaData,
                        String, Table, delete, insert, select, update)
from sqlalchemy.exc import IntegrityError


def _hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    derived = hashlib.scrypt(password.encode('utf-8'), salt=bytes.fromhex(salt),
                             n=16384, r=8, p=1, dklen=32)
    return 'scrypt$' + salt + '$' + derived.hex()


def _verify(password, encoded):
    return hmac.compare_digest(_hash_password(password, encoded.split('$')[1]), encoded)


_DUMMY_PASSWORD = _hash_password('unusable-dummy-password')
_USER_FIELDS = ('id', 'username', 'display_name', 'tenant_id', 'role',
                'dataset_id', 'visibility', 'enabled')


def _public(row):
    return {key: row[key] for key in _USER_FIELDS}


class AuthStore:
    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData()
        self.users = Table('kf_users', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('username', String(128), nullable=False, unique=True),
            Column('password_hash', String(160), nullable=False),
            Column('display_name', String(128), nullable=False),
            Column('tenant_id', String(128), nullable=False, index=True),
            Column('role', String(32), nullable=False),
            Column('dataset_id', String(128), nullable=False),
            Column('visibility', String(32), nullable=False),
            Column('enabled', Boolean, nullable=False, default=True))
        self.tokens = Table('kf_login_tokens', self.metadata,
            Column('token_hash', String(64), primary_key=True),
            Column('user_id', Integer, ForeignKey('kf_users.id'), nullable=False, index=True),
            Column('csrf_token', String(64), nullable=False),
            Column('expires_at', DateTime, nullable=False))
        self.sessions = Table('kf_chat_sessions', self.metadata,
            Column('id', String(36), primary_key=True),
            Column('user_id', Integer, ForeignKey('kf_users.id'), nullable=False, index=True),
            Column('tenant_id', String(128), nullable=False, index=True),
            Column('dataset_id', String(128), nullable=False),
            Column('title', String(200), nullable=False),
            Column('created_at', DateTime, nullable=False))

    def initialize(self):
        self.metadata.create_all(self.engine)

    def create_user(self, username, password, display_name, tenant_id,
                    role='viewer', dataset_id='default', visibility='public'):
        if visibility not in {'public', 'internal', 'private'}:
            raise ValueError('Invalid visibility')
        username = username.strip().lower()
        if not 12 <= len(password) <= 256:
            raise ValueError('Password must contain 12 to 256 characters')
        if role not in {'viewer', 'knowledge_admin', 'tenant_admin'}:
            raise ValueError('Invalid role')
        for name, value, maximum in [('username', username, 128),
                ('display_name', display_name, 128), ('tenant_id', tenant_id, 128),
                ('dataset_id', dataset_id, 128), ('visibility', visibility, 32)]:
            if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                raise ValueError('Invalid ' + name)
        values = dict(username=username, password_hash=_hash_password(password),
                      display_name=display_name, tenant_id=tenant_id, role=role,
                      dataset_id=dataset_id, visibility=visibility, enabled=True)
        try:
            with self.engine.begin() as conn:
                result = conn.execute(insert(self.users).values(**values))
                values['id'] = result.inserted_primary_key[0]
        except IntegrityError as exc:
            raise ValueError('Username already exists') from exc
        return _public(values)

    def login(self, username, password):
        # Always derive one password hash, even for nonexistent/disabled users.
        with self.engine.begin() as conn:
            row = conn.execute(select(self.users).where(
                self.users.c.username == username.strip().lower()).with_for_update()).mappings().first()
            candidate = password if isinstance(password, str) and len(password) <= 256 else ''
            valid = _verify(candidate, row['password_hash'] if row else _DUMMY_PASSWORD)
            if not valid or not row or not row['enabled']:
                raise ValueError('Invalid credentials')
            token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            conn.execute(insert(self.tokens).values(token_hash=self._token_hash(token),
                user_id=row['id'], csrf_token=csrf,
                expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)))
            return token, csrf, _public(row)

    @staticmethod
    def _token_hash(token):
        return hashlib.sha256(token.encode('utf-8')).hexdigest()

    def authenticate(self, raw_token):
        if not raw_token or not isinstance(raw_token, str) or len(raw_token) > 256:
            return None
        with self.engine.connect() as conn:
            row = conn.execute(select(self.users, self.tokens.c.csrf_token).select_from(
                self.users.join(self.tokens, self.tokens.c.user_id == self.users.c.id)).where(
                    self.tokens.c.token_hash == self._token_hash(raw_token),
                    self.tokens.c.expires_at > datetime.now(timezone.utc).replace(tzinfo=None),
                    self.users.c.enabled.is_(True))).mappings().first()
        return dict(_public(row), csrf_token=row['csrf_token']) if row else None

    def logout(self, token):
        if token:
            with self.engine.begin() as conn:
                conn.execute(delete(self.tokens).where(self.tokens.c.token_hash == self._token_hash(token)))

    def list_users(self, tenant_id):
        with self.engine.connect() as conn:
            return [_public(row) for row in conn.execute(select(self.users).where(
                self.users.c.tenant_id == tenant_id).order_by(self.users.c.id)).mappings()]

    def set_enabled(self, actor_tenant_id, user_id, enabled):
        with self.engine.begin() as conn:
            row = conn.execute(select(self.users.c.id).where(self.users.c.id == user_id,
                self.users.c.tenant_id == actor_tenant_id).with_for_update()).first()
            if row is None:
                return False
            conn.execute(update(self.users).where(self.users.c.id == user_id).values(enabled=enabled))
            if not enabled:
                conn.execute(delete(self.tokens).where(self.tokens.c.user_id == user_id))
        return True

    def create_session(self, user_id, tenant_id, title='新会话'):
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise ValueError('Invalid session title')
        with self.engine.begin() as conn:
            row = conn.execute(select(self.users).where(self.users.c.id == user_id,
                self.users.c.tenant_id == tenant_id, self.users.c.enabled.is_(True))).mappings().first()
            if row is None:
                raise ValueError('Invalid session owner')
            values = dict(id=str(uuid.uuid4()), user_id=user_id, tenant_id=tenant_id,
                          dataset_id=row['dataset_id'], title=title.strip(), created_at=datetime.now(timezone.utc).replace(tzinfo=None))
            conn.execute(insert(self.sessions).values(**values))
        return self._session_public(values)

    @staticmethod
    def _session_public(row):
        result = dict(row)
        result['created_at'] = result['created_at'].isoformat() + 'Z'
        return result

    def _owned(self, user_id, tenant_id):
        return (self.sessions.c.user_id == user_id, self.sessions.c.tenant_id == tenant_id)

    def list_sessions(self, user_id, tenant_id):
        with self.engine.connect() as conn:
            rows = conn.execute(select(self.sessions).where(*self._owned(user_id, tenant_id))
                .order_by(self.sessions.c.created_at.desc())).mappings()
            return [self._session_public(row) for row in rows]

    def get_session(self, session_id, user_id, tenant_id):
        with self.engine.connect() as conn:
            row = conn.execute(select(self.sessions).where(self.sessions.c.id == session_id,
                *self._owned(user_id, tenant_id))).mappings().first()
            return self._session_public(row) if row else None

    def delete_session(self, session_id, user_id, tenant_id):
        with self.engine.begin() as conn:
            return bool(conn.execute(delete(self.sessions).where(self.sessions.c.id == session_id,
                *self._owned(user_id, tenant_id))).rowcount)
