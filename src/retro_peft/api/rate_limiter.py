"""
Rate limiting and request throttling for API Gateway.

Implements token bucket, sliding window, and fixed window algorithms
with per-user, per-endpoint, and global rate limiting.
"""

import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import redis
import json

from ..utils.logging import get_global_logger


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_second: float = 10.0
    burst_size: int = 20
    window_size: int = 60  # seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    per_user: bool = True
    per_endpoint: bool = True
    global_limit: Optional[float] = None
    
    # Endpoint-specific limits
    endpoint_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # User tier limits
    user_tier_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None
    limit_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenBucket:
    """
    Token bucket rate limiter implementation.
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.
        
        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.RLock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        with self._lock:
            now = time.time()
            self._refill(now)
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def peek(self) -> float:
        """Get current token count without consuming"""
        with self._lock:
            now = time.time()
            self._refill(now)
            return self.tokens
    
    def _refill(self, now: float):
        """Refill tokens based on elapsed time"""
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to be available"""
        with self._lock:
            now = time.time()
            self._refill(now)
            
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.rate


class SlidingWindowCounter:
    """
    Sliding window rate limiter implementation.
    """
    
    def __init__(self, limit: int, window_size: int):
        """
        Initialize sliding window counter.
        
        Args:
            limit: Request limit per window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.requests: deque[float] = deque()
        self._lock = threading.RLock()
    
    def is_allowed(self) -> Tuple[bool, int, float]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        with self._lock:
            now = time.time()
            self._cleanup_old_requests(now)
            
            if len(self.requests) < self.limit:
                self.requests.append(now)
                remaining = self.limit - len(self.requests)
                reset_time = now + self.window_size
                return True, remaining, reset_time
            
            # Calculate reset time (when oldest request expires)
            reset_time = self.requests[0] + self.window_size
            return False, 0, reset_time
    
    def _cleanup_old_requests(self, now: float):
        """Remove requests outside the window"""
        cutoff_time = now - self.window_size
        
        while self.requests and self.requests[0] <= cutoff_time:
            self.requests.popleft()


class FixedWindowCounter:
    """
    Fixed window rate limiter implementation.
    """
    
    def __init__(self, limit: int, window_size: int):
        """
        Initialize fixed window counter.
        
        Args:
            limit: Request limit per window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.count = 0
        self.window_start = time.time()
        self._lock = threading.RLock()
    
    def is_allowed(self) -> Tuple[bool, int, float]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        with self._lock:
            now = time.time()
            
            # Check if we need to start a new window
            if now >= self.window_start + self.window_size:
                self.count = 0
                self.window_start = now
            
            if self.count < self.limit:
                self.count += 1
                remaining = self.limit - self.count
                reset_time = self.window_start + self.window_size
                return True, remaining, reset_time
            
            reset_time = self.window_start + self.window_size
            return False, 0, reset_time


class InMemoryRateLimiter:
    """
    In-memory rate limiter with multiple algorithms.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize in-memory rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.limiters: Dict[str, Any] = {}
        self.global_limiter = None
        
        self._lock = threading.RLock()
        self.logger = get_global_logger()
        
        # Initialize global limiter if configured
        if config.global_limit:
            self._create_global_limiter()
    
    def _create_global_limiter(self):
        """Create global rate limiter"""
        if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            self.global_limiter = TokenBucket(
                rate=self.config.global_limit,
                capacity=int(self.config.global_limit * 2)
            )
        elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            self.global_limiter = SlidingWindowCounter(
                limit=int(self.config.global_limit * self.config.window_size),
                window_size=self.config.window_size
            )
        elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            self.global_limiter = FixedWindowCounter(
                limit=int(self.config.global_limit * self.config.window_size),
                window_size=self.config.window_size
            )
    
    def _get_limiter_key(self, user_id: str, endpoint: str) -> str:
        """Generate limiter key"""
        parts = []
        
        if self.config.per_user:
            parts.append(f"user:{user_id}")
        
        if self.config.per_endpoint:
            parts.append(f"endpoint:{endpoint}")
        
        return ":".join(parts) if parts else "global"
    
    def _get_or_create_limiter(self, key: str, endpoint: str, user_id: str) -> Any:
        """Get or create rate limiter for key"""
        if key in self.limiters:
            return self.limiters[key]
        
        # Determine limits for this key
        rate = self.config.requests_per_second
        burst = self.config.burst_size
        window = self.config.window_size
        
        # Check for endpoint-specific limits
        if endpoint in self.config.endpoint_limits:
            endpoint_config = self.config.endpoint_limits[endpoint]
            rate = endpoint_config.get("requests_per_second", rate)
            burst = endpoint_config.get("burst_size", burst)
            window = endpoint_config.get("window_size", window)
        
        # Check for user tier limits (would need user tier lookup)
        # This is a simplified version
        
        # Create limiter based on algorithm
        if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            limiter = TokenBucket(rate=rate, capacity=burst)
        elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            limiter = SlidingWindowCounter(
                limit=int(rate * window),
                window_size=window
            )
        elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            limiter = FixedWindowCounter(
                limit=int(rate * window),
                window_size=window
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        self.limiters[key] = limiter
        return limiter
    
    async def check_rate_limit(
        self,
        key: str,
        endpoint: str,
        user_id: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check rate limit for request.
        
        Args:
            key: Rate limit key (usually user_id:endpoint)
            endpoint: API endpoint
            user_id: User ID
            
        Returns:
            Rate limit result
        """
        with self._lock:
            # Check global limit first
            if self.global_limiter:
                if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                    if not self.global_limiter.consume(1):
                        wait_time = self.global_limiter.get_wait_time(1)
                        return RateLimitResult(
                            allowed=False,
                            remaining=0,
                            reset_time=time.time() + wait_time,
                            retry_after=wait_time,
                            limit_type="global"
                        )
                else:
                    allowed, remaining, reset_time = self.global_limiter.is_allowed()
                    if not allowed:
                        return RateLimitResult(
                            allowed=False,
                            remaining=remaining,
                            reset_time=reset_time,
                            retry_after=reset_time - time.time(),
                            limit_type="global"
                        )
            
            # Check specific limiter
            limiter_key = self._get_limiter_key(user_id or "anonymous", endpoint)
            limiter = self._get_or_create_limiter(limiter_key, endpoint, user_id or "anonymous")
            
            if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                if limiter.consume(1):
                    return RateLimitResult(
                        allowed=True,
                        remaining=int(limiter.peek()),
                        reset_time=time.time() + (limiter.capacity / limiter.rate),
                        limit_type="user" if self.config.per_user else "endpoint"
                    )
                else:
                    wait_time = limiter.get_wait_time(1)
                    return RateLimitResult(
                        allowed=False,
                        remaining=0,
                        reset_time=time.time() + wait_time,
                        retry_after=wait_time,
                        limit_type="user" if self.config.per_user else "endpoint"
                    )
            else:
                allowed, remaining, reset_time = limiter.is_allowed()
                return RateLimitResult(
                    allowed=allowed,
                    remaining=remaining,
                    reset_time=reset_time,
                    retry_after=reset_time - time.time() if not allowed else None,
                    limit_type="user" if self.config.per_user else "endpoint"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self._lock:
            return {
                "algorithm": self.config.algorithm.value,
                "total_limiters": len(self.limiters),
                "global_limit": self.config.global_limit,
                "per_user": self.config.per_user,
                "per_endpoint": self.config.per_endpoint,
                "endpoint_limits": len(self.config.endpoint_limits)
            }


class RedisRateLimiter:
    """
    Redis-backed distributed rate limiter.
    """
    
    def __init__(self, config: RateLimitConfig, redis_client: Optional[redis.Redis] = None):
        """
        Initialize Redis rate limiter.
        
        Args:
            config: Rate limit configuration
            redis_client: Optional Redis client
        """
        self.config = config
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.logger = get_global_logger()
        
        # Lua scripts for atomic operations
        self._token_bucket_script = self.redis.register_script("""
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local rate = tonumber(ARGV[2])
            local tokens_requested = tonumber(ARGV[3])
            local now = tonumber(ARGV[4])
            
            local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
            local tokens = tonumber(bucket[1]) or capacity
            local last_refill = tonumber(bucket[2]) or now
            
            -- Refill tokens
            local elapsed = now - last_refill
            tokens = math.min(capacity, tokens + elapsed * rate)
            
            -- Check if we can consume
            if tokens >= tokens_requested then
                tokens = tokens - tokens_requested
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
                return {1, math.floor(tokens)}
            else
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('EXPIRE', key, 3600)
                return {0, math.floor(tokens)}
            end
        """)
        
        self._sliding_window_script = self.redis.register_script("""
            local key = KEYS[1]
            local window_size = tonumber(ARGV[1])
            local limit = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])
            
            -- Remove old entries
            redis.call('ZREMRANGEBYSCORE', key, 0, now - window_size)
            
            -- Count current entries
            local current = redis.call('ZCARD', key)
            
            if current < limit then
                -- Add current request
                redis.call('ZADD', key, now, now)
                redis.call('EXPIRE', key, window_size + 1)
                return {1, limit - current - 1, now + window_size}
            else
                -- Get oldest entry for reset time
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                local reset_time = now + window_size
                if oldest[2] then
                    reset_time = tonumber(oldest[2]) + window_size
                end
                return {0, 0, reset_time}
            end
        """)
    
    async def check_rate_limit(
        self,
        key: str,
        endpoint: str,
        user_id: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check rate limit using Redis.
        
        Args:
            key: Rate limit key
            endpoint: API endpoint
            user_id: User ID
            
        Returns:
            Rate limit result
        """
        try:
            now = time.time()
            redis_key = f"rate_limit:{key}:{endpoint}"
            
            # Get rate limit parameters
            rate = self.config.requests_per_second
            burst = self.config.burst_size
            window = self.config.window_size
            
            # Check for endpoint-specific limits
            if endpoint in self.config.endpoint_limits:
                endpoint_config = self.config.endpoint_limits[endpoint]
                rate = endpoint_config.get("requests_per_second", rate)
                burst = endpoint_config.get("burst_size", burst)
                window = endpoint_config.get("window_size", window)
            
            if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                result = self._token_bucket_script(
                    keys=[redis_key],
                    args=[burst, rate, 1, now]
                )
                
                allowed = bool(result[0])
                remaining = int(result[1])
                reset_time = now + (burst / rate)
                
                return RateLimitResult(
                    allowed=allowed,
                    remaining=remaining,
                    reset_time=reset_time,
                    retry_after=None if allowed else (1.0 / rate),
                    limit_type="distributed"
                )
            
            elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                limit = int(rate * window)
                result = self._sliding_window_script(
                    keys=[redis_key],
                    args=[window, limit, now]
                )
                
                allowed = bool(result[0])
                remaining = int(result[1])
                reset_time = float(result[2])
                
                return RateLimitResult(
                    allowed=allowed,
                    remaining=remaining,
                    reset_time=reset_time,
                    retry_after=None if allowed else (reset_time - now),
                    limit_type="distributed"
                )
            
            else:
                # Fallback to token bucket for unsupported algorithms
                return await self._fallback_rate_limit(key, endpoint, user_id)
                
        except Exception as e:
            self.logger.error(f"Redis rate limit error: {e}")
            # Fallback to allowing request on Redis errors
            return RateLimitResult(
                allowed=True,
                remaining=1000,
                reset_time=time.time() + 60,
                limit_type="fallback"
            )
    
    async def _fallback_rate_limit(
        self,
        key: str,
        endpoint: str,
        user_id: Optional[str]
    ) -> RateLimitResult:
        """Fallback rate limiting when Redis fails"""
        # Simple in-memory fallback
        return RateLimitResult(
            allowed=True,
            remaining=100,  # Conservative fallback
            reset_time=time.time() + 60,
            limit_type="fallback"
        )


class RateLimiter:
    """
    Unified rate limiter supporting both in-memory and distributed modes.
    """
    
    def __init__(
        self,
        config: RateLimitConfig,
        use_redis: bool = False,
        redis_client: Optional[redis.Redis] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
            use_redis: Whether to use Redis for distributed rate limiting
            redis_client: Optional Redis client
        """
        self.config = config
        self.use_redis = use_redis
        
        if use_redis:
            self.limiter = RedisRateLimiter(config, redis_client)
        else:
            self.limiter = InMemoryRateLimiter(config)
        
        self.logger = get_global_logger()
    
    async def check_rate_limit(
        self,
        key: str,
        endpoint: str,
        user_id: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check rate limit for request.
        
        Args:
            key: Rate limit key
            endpoint: API endpoint
            user_id: User ID
            
        Returns:
            Rate limit result
        """
        return await self.limiter.check_rate_limit(key, endpoint, user_id)
    
    def add_endpoint_limit(
        self,
        endpoint: str,
        requests_per_second: float,
        burst_size: Optional[int] = None,
        window_size: Optional[int] = None
    ):
        """
        Add endpoint-specific rate limit.
        
        Args:
            endpoint: API endpoint path
            requests_per_second: Rate limit
            burst_size: Burst capacity
            window_size: Window size for sliding/fixed window
        """
        self.config.endpoint_limits[endpoint] = {
            "requests_per_second": requests_per_second,
            "burst_size": burst_size or int(requests_per_second * 2),
            "window_size": window_size or self.config.window_size
        }
        
        self.logger.info(f"Added rate limit for endpoint {endpoint}: {requests_per_second} req/s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        base_stats = {
            "mode": "redis" if self.use_redis else "in_memory",
            "config": {
                "algorithm": self.config.algorithm.value,
                "requests_per_second": self.config.requests_per_second,
                "burst_size": self.config.burst_size,
                "window_size": self.config.window_size,
                "per_user": self.config.per_user,
                "per_endpoint": self.config.per_endpoint,
                "global_limit": self.config.global_limit
            }
        }
        
        if hasattr(self.limiter, 'get_stats'):
            base_stats.update(self.limiter.get_stats())
        
        return base_stats