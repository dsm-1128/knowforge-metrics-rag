"""缓存存储适配器：L1 进程内存缓存和 L2 Redis 缓存。

设计决策：
- TTLMemoryCache：使用字典存储，值带过期时间戳。每次 get 时惰性删除过期条目。
  适合存储 namespace epoch 等低频请求的元数据（TTL 通常 3s），不需要后台
  清理线程或定时器。
- RedisJsonCache：使用 Python redis 客户端连接 Redis。decode_responses=True
  让客户端直接返回字符串而非 bytes，减少 get_json 中的解码步骤。

调用顺序：CacheManager -> stores。
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from typing import Any

from qa_core.config.settings import get_settings


@dataclass
class _MemoryItem:
    """进程内存缓存的条目，包含值和过期时间。

    调用顺序：TTLMemoryCache -> _MemoryItem。
    """

    value: Any  # 字段说明：缓存的值
    expires_at: float  # 字段说明：过期时间戳（time.time() 值，单位秒）


class TTLMemoryCache:
    """进程内短 TTL 缓存，用于 cache namespace epoch 等低频元数据。

    这是一个简单的字典 + 惰性过期实现的缓存。每次 get 时检查条目是否过期，
    过期后自动移除并返回 None。不支持 TTL 续期、淘汰策略或容量限制——
    设计使用场景是 namespace epoch 等少量且变化不频繁的元数据。

    调用顺序：CacheManager -> TTLMemoryCache。
    """

    def __init__(self) -> None:
        """初始化进程内存缓存。

        调用顺序：CacheManager.__init__() -> TTLMemoryCache.__init__()。
        """
        self._items: dict[str, _MemoryItem] = {}

    def get(self, key: str) -> Any | None:
        """读取缓存键；未命中或过期时返回 None。

        检查条目过期时间：如果 expires_at <= 当前时间，说明条目已过期，删除后
        返回 None。过期的条目不会在这里被后台线程清理——惰性删除足够高效，
        因为每个过期条目最多在内存中多停留一个 get 周期。

        参数：
            key: 缓存键。

        返回：
            缓存的值；未命中或已过期时返回 None。

        调用顺序：CacheManager.namespace_epoch() 等 -> TTLMemoryCache.get()。
        """
        item = self._items.get(key)
        if item is None:
            return None
        # 惰性过期检查：仅在读取时判断是否过期，没有后台清理线程
        if item.expires_at <= time.time():
            self._items.pop(key, None)
            return None
        return item.value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """写入缓存值并应用调用方提供的过期时间。

        参数：
            key: 缓存键。
            value: 缓存的值。
            ttl_seconds: 过期时间（秒），最小为 1 秒。

        调用顺序：CacheManager.namespace_epoch() 等 -> TTLMemoryCache.set()。
        """
        # max(1, ttl_seconds) 保证 TTL 至少 1 秒，避免传入 0 或负数导致条目立即过期
        self._items[key] = _MemoryItem(value=value, expires_at=time.time() + max(1, ttl_seconds))

    def clear(self) -> None:
        """清空当前进程缓存，供版本切换和测试隔离使用。

        调用顺序：CacheManager.invalidate_scenario() 等 -> TTLMemoryCache.clear()。
        """
        self._items.clear()


class RedisJsonCache:
    """Redis JSON 缓存适配器，封装 get_json/set_json/delete_pattern/ping 操作。

    使用 decode_responses=True 让 Redis 客户端直接返回字符串而非 bytes，
    减少序列化层的类型转换开销。socket_timeout 和 socket_connect_timeout 统一
    从 settings 读取，避免连接 hang 导致请求卡死。

    调用顺序：CacheManager -> RedisJsonCache。
    """

    def __init__(self, client: Any | None = None) -> None:
        """初始化 Redis 缓存适配器。

        支持注入 mock client 用于测试。未注入时根据 settings.redis_url 动态
        创建连接。redis Python 包是可选依赖，在 import 时捕获 ImportError。

        参数：
            client: 可选的 Redis 客户端实例，注入后跳过自动创建。

        调用顺序：CacheManager._redis() -> RedisJsonCache.__init__()。
        """
        settings = get_settings()
        if client is not None:
            # 原因：注入 mock client，测试时不需要真实 Redis 连接
            self.client = client
            return
        # 原因：redis 是可选依赖，不在 requirements.txt 中强制安装
        try:
            redis_module = importlib.import_module("redis")
        except ImportError as exc:
            raise RuntimeError("Redis 缓存已启用，但未安装 redis 依赖。请安装 requirements.txt。") from exc
        self.client = redis_module.Redis.from_url(
            settings.redis_url,
            # decode_responses=True：让客户端直接返回字符串而非 bytes
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
        )

    def get_json(self, key: str) -> dict[str, Any] | None:
        """读取并反序列化 JSON 缓存值。

        参数：
            key: Redis 键。

        返回：
            反序列化后的字典；key 不存在时返回 None。

        调用顺序：CacheManager.get_retrieval_result() / get_embedding_vector() -> RedisJsonCache.get_json()。
        """
        raw = self.client.get(key)
        if not raw:
            return None
        return json.loads(raw)

    def set_json(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        """序列化对象并写入带过期时间的 JSON 缓存。

        ensure_ascii=False 保证中文等非 ASCII 字符以原始形式存储，便于
        在 Redis CLI 中直接查看。

        参数：
            key: Redis 键。
            payload: 待序列化的字典。
            ttl_seconds: 过期时间（秒），最小为 1 秒。

        调用顺序：CacheManager.set_retrieval_result() / set_embedding_vector() -> RedisJsonCache.set_json()。
        """
        # max(1, int(ttl_seconds)) 保证 TTL 至少 1 秒且为整数，避免 Redis 报错
        self.client.set(key, json.dumps(payload, ensure_ascii=False), ex=max(1, int(ttl_seconds)))

    def delete_pattern(self, pattern: str) -> int:
        """按 pattern 批量删除 Redis key。（★★ 理解）

        使用 scan_iter 而非 keys()：scan_iter 分批迭代（count=500），不会阻塞
        Redis 主线程。keys() 在 key 数量大时可能导致 Redis 短暂不可用。

        目前的缓存失效策略是 namespace epoch 方式（旧 key 自然耗尽 TTL），
        该方法主要用于管理端手动清理或测试后的数据清除。

        参数：
            pattern: Redis key 匹配模式（如 "kf:v1:*"）。

        返回：
            被删除的 key 数量。

        调用顺序：管理接口或测试 -> RedisJsonCache.delete_pattern()。
        """
        # 原因：scan_iter 分批迭代，count=500 控制每次迭代返回的 key 数量
        keys = list(self.client.scan_iter(match=pattern, count=500))
        if not keys:
            return 0
        return int(self.client.delete(*keys))

    def ping(self) -> bool:
        """检查缓存后端连接是否可用。

        使用 Redis PING 命令快速检测连接状态。返回 True 表示连接正常。
        此方法不会抛出异常——由调用方负责处理连接失败场景。

        返回：
            True：连接正常；False：连接异常。

        调用顺序：CacheManager.status() -> RedisJsonCache.ping()。
        """
        return bool(self.client.ping())
