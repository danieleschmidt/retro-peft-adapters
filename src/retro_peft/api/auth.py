"""
Authentication and authorization for API Gateway.

Supports JWT tokens, API keys, and role-based access control.
"""

import hashlib
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import jwt
from passlib.context import CryptContext

from ..utils.logging import get_global_logger


@dataclass
class AuthResult:
    """Authentication result"""

    success: bool
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """User information"""

    id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAuthenticator(ABC):
    """Base class for authenticators"""

    @abstractmethod
    async def authenticate(self, credentials: str) -> AuthResult:
        """Authenticate user with credentials"""
        pass

    @abstractmethod
    async def validate_permissions(self, user_id: str, required_permissions: List[str]) -> bool:
        """Validate user permissions"""
        pass


class JWTAuthenticator(BaseAuthenticator):
    """
    JWT-based authentication with role-based access control.
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        token_expiry: int = 3600,  # 1 hour
        issuer: str = "retro-peft-api",
    ):
        """
        Initialize JWT authenticator.

        Args:
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm
            token_expiry: Token expiry time in seconds
            issuer: JWT issuer
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = token_expiry
        self.issuer = issuer

        # In-memory user store (would be replaced with database)
        self.users: Dict[str, User] = {}
        self.user_tokens: Dict[str, str] = {}  # user_id -> current_token

        self.logger = get_global_logger()

    def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        roles: List[str] = None,
        permissions: List[str] = None,
    ) -> User:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            roles: User roles
            permissions: User permissions

        Returns:
            Created user
        """
        user_id = hashlib.sha256(f"{username}_{time.time()}".encode()).hexdigest()[:16]

        user = User(
            id=user_id,
            username=username,
            email=email,
            roles=roles or [],
            permissions=permissions or [],
        )

        self.users[user_id] = user
        self.logger.info(f"Created user: {username} (ID: {user_id})")

        return user

    def create_token(self, user_id: str) -> str:
        """
        Create JWT token for user.

        Args:
            user_id: User ID

        Returns:
            JWT token
        """
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")

        user = self.users[user_id]
        now = time.time()

        payload = {
            "sub": user_id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "iat": int(now),
            "exp": int(now + self.token_expiry),
            "iss": self.issuer,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        self.user_tokens[user_id] = token

        # Update last login
        user.last_login = now

        return token

    async def authenticate(self, credentials: str) -> AuthResult:
        """
        Authenticate JWT token.

        Args:
            credentials: Authorization header value

        Returns:
            Authentication result
        """
        try:
            # Extract token from "Bearer <token>"
            if not credentials.startswith("Bearer "):
                return AuthResult(success=False, error="Invalid authorization format")

            token = credentials[7:]  # Remove "Bearer "

            # Decode and validate token
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm], issuer=self.issuer
            )

            user_id = payload.get("sub")
            if not user_id:
                return AuthResult(success=False, error="Invalid token payload")

            # Check if user exists and is active
            if user_id not in self.users:
                return AuthResult(success=False, error="User not found")

            user = self.users[user_id]
            if not user.is_active:
                return AuthResult(success=False, error="User is inactive")

            # Check token freshness (optional revocation check)
            current_token = self.user_tokens.get(user_id)
            if current_token and current_token != token:
                return AuthResult(success=False, error="Token has been revoked")

            return AuthResult(
                success=True,
                user_id=user_id,
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                metadata={"username": payload.get("username"), "token_exp": payload.get("exp")},
            )

        except jwt.ExpiredSignatureError:
            return AuthResult(success=False, error="Token has expired")
        except jwt.InvalidTokenError as e:
            return AuthResult(success=False, error=f"Invalid token: {e}")
        except Exception as e:
            self.logger.error(f"JWT authentication error: {e}")
            return AuthResult(success=False, error="Authentication failed")

    async def validate_permissions(self, user_id: str, required_permissions: List[str]) -> bool:
        """
        Validate user permissions.

        Args:
            user_id: User ID
            required_permissions: List of required permissions

        Returns:
            True if user has all required permissions
        """
        if user_id not in self.users:
            return False

        user = self.users[user_id]
        user_permissions = set(user.permissions)

        # Check if user has all required permissions
        return all(perm in user_permissions for perm in required_permissions)

    def revoke_token(self, user_id: str):
        """Revoke user's current token"""
        if user_id in self.user_tokens:
            del self.user_tokens[user_id]
            self.logger.info(f"Revoked token for user {user_id}")


class APIKeyAuthenticator(BaseAuthenticator):
    """
    API key-based authentication.
    """

    def __init__(self):
        self.api_keys: Dict[str, User] = {}  # api_key -> user
        self.key_usage: Dict[str, Dict[str, Any]] = {}  # api_key -> usage stats
        self.logger = get_global_logger()

    def create_api_key(self, user: User, key_name: Optional[str] = None) -> str:
        """
        Create API key for user.

        Args:
            user: User object
            key_name: Optional key name

        Returns:
            Generated API key
        """
        # Generate secure random key
        api_key = f"retro_{secrets.token_urlsafe(32)}"

        # Store key
        self.api_keys[api_key] = user
        user.api_keys.append(api_key)

        # Initialize usage tracking
        self.key_usage[api_key] = {
            "created_at": time.time(),
            "name": key_name or f"key_{len(user.api_keys)}",
            "total_requests": 0,
            "last_used": None,
        }

        self.logger.info(f"Created API key for user {user.username}")
        return api_key

    async def authenticate(self, credentials: str) -> AuthResult:
        """
        Authenticate API key.

        Args:
            credentials: Authorization header value or direct API key

        Returns:
            Authentication result
        """
        try:
            # Extract API key
            api_key = credentials
            if credentials.startswith("Bearer "):
                api_key = credentials[7:]
            elif credentials.startswith("API-Key "):
                api_key = credentials[8:]

            # Validate key format
            if not api_key.startswith("retro_"):
                return AuthResult(success=False, error="Invalid API key format")

            # Check if key exists
            if api_key not in self.api_keys:
                return AuthResult(success=False, error="Invalid API key")

            user = self.api_keys[api_key]

            # Check if user is active
            if not user.is_active:
                return AuthResult(success=False, error="User is inactive")

            # Update usage stats
            self.key_usage[api_key]["total_requests"] += 1
            self.key_usage[api_key]["last_used"] = time.time()

            return AuthResult(
                success=True,
                user_id=user.id,
                roles=user.roles,
                permissions=user.permissions,
                metadata={
                    "username": user.username,
                    "api_key": api_key[:10] + "...",  # Partial key for logging
                    "key_name": self.key_usage[api_key]["name"],
                },
            )

        except Exception as e:
            self.logger.error(f"API key authentication error: {e}")
            return AuthResult(success=False, error="Authentication failed")

    async def validate_permissions(self, user_id: str, required_permissions: List[str]) -> bool:
        """
        Validate user permissions.

        Args:
            user_id: User ID
            required_permissions: List of required permissions

        Returns:
            True if user has all required permissions
        """
        # Find user by ID
        user = None
        for api_user in self.api_keys.values():
            if api_user.id == user_id:
                user = api_user
                break

        if not user:
            return False

        user_permissions = set(user.permissions)
        return all(perm in user_permissions for perm in required_permissions)

    def revoke_api_key(self, api_key: str):
        """Revoke API key"""
        if api_key in self.api_keys:
            user = self.api_keys[api_key]
            del self.api_keys[api_key]
            del self.key_usage[api_key]

            # Remove from user's key list
            if api_key in user.api_keys:
                user.api_keys.remove(api_key)

            self.logger.info(f"Revoked API key for user {user.username}")

    def get_key_usage(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for API key"""
        return self.key_usage.get(api_key)


class AuthenticationManager:
    """
    Central authentication manager supporting multiple auth methods.
    """

    def __init__(self):
        self.authenticators: Dict[str, BaseAuthenticator] = {}
        self.default_permissions = ["read"]
        self.admin_permissions = ["read", "write", "admin"]
        self.logger = get_global_logger()

    def add_authenticator(self, name: str, authenticator: BaseAuthenticator):
        """Add authenticator"""
        self.authenticators[name] = authenticator
        self.logger.info(f"Added authenticator: {name}")

    async def authenticate(self, credentials: str) -> AuthResult:
        """
        Authenticate using available authenticators.

        Args:
            credentials: Authorization header value

        Returns:
            Authentication result
        """
        # Try each authenticator until one succeeds
        for name, authenticator in self.authenticators.items():
            try:
                result = await authenticator.authenticate(credentials)
                if result.success:
                    result.metadata["auth_method"] = name
                    return result
            except Exception as e:
                self.logger.error(f"Authenticator {name} failed: {e}")
                continue

        return AuthResult(success=False, error="Authentication failed")

    async def validate_permissions(
        self, user_id: str, required_permissions: List[str], auth_method: Optional[str] = None
    ) -> bool:
        """
        Validate user permissions.

        Args:
            user_id: User ID
            required_permissions: Required permissions
            auth_method: Optional specific auth method to use

        Returns:
            True if user has required permissions
        """
        authenticators_to_check = []

        if auth_method and auth_method in self.authenticators:
            authenticators_to_check.append(self.authenticators[auth_method])
        else:
            authenticators_to_check = list(self.authenticators.values())

        for authenticator in authenticators_to_check:
            try:
                if await authenticator.validate_permissions(user_id, required_permissions):
                    return True
            except Exception as e:
                self.logger.error(f"Permission validation error: {e}")
                continue

        return False

    def create_default_users(self) -> Dict[str, str]:
        """
        Create default admin and demo users.

        Returns:
            Dictionary of username -> token/api_key
        """
        credentials = {}

        # Create JWT authenticator if not exists
        if "jwt" not in self.authenticators:
            jwt_auth = JWTAuthenticator(secret_key=secrets.token_urlsafe(32))
            self.add_authenticator("jwt", jwt_auth)

        # Create API key authenticator if not exists
        if "api_key" not in self.authenticators:
            api_key_auth = APIKeyAuthenticator()
            self.add_authenticator("api_key", api_key_auth)

        jwt_auth = self.authenticators["jwt"]
        api_key_auth = self.authenticators["api_key"]

        # Create admin user
        admin_user = jwt_auth.create_user(
            username="admin",
            email="admin@retro-peft.com",
            roles=["admin"],
            permissions=self.admin_permissions,
        )

        admin_token = jwt_auth.create_token(admin_user.id)
        admin_api_key = api_key_auth.create_api_key(admin_user, "admin_key")

        credentials["admin_jwt"] = admin_token
        credentials["admin_api_key"] = admin_api_key

        # Create demo user
        demo_user = jwt_auth.create_user(
            username="demo",
            email="demo@retro-peft.com",
            roles=["user"],
            permissions=self.default_permissions,
        )

        demo_token = jwt_auth.create_token(demo_user.id)
        demo_api_key = api_key_auth.create_api_key(demo_user, "demo_key")

        credentials["demo_jwt"] = demo_token
        credentials["demo_api_key"] = demo_api_key

        self.logger.info("Created default admin and demo users")
        return credentials

    def get_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        stats = {
            "authenticators": list(self.authenticators.keys()),
            "default_permissions": self.default_permissions,
            "admin_permissions": self.admin_permissions,
        }

        # Add JWT stats if available
        if "jwt" in self.authenticators:
            jwt_auth = self.authenticators["jwt"]
            stats["jwt"] = {
                "total_users": len(jwt_auth.users),
                "active_tokens": len(jwt_auth.user_tokens),
            }

        # Add API key stats if available
        if "api_key" in self.authenticators:
            api_key_auth = self.authenticators["api_key"]
            stats["api_key"] = {
                "total_keys": len(api_key_auth.api_keys),
                "total_requests": sum(
                    usage["total_requests"] for usage in api_key_auth.key_usage.values()
                ),
            }

        return stats
