"""Authenticating a caller and turning it into a Principal."""

import time
from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException, status

from llmapp.rag.access import Principal

ALGORITHM = "HS256"


class AuthError(HTTPException):
    """401 with no hint about why."""

    def __init__(self, detail: str = "not authenticated") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


@dataclass(frozen=True)
class TokenIssuer:
    """Issues and verifies the tokens this service accepts."""

    secret: str
    ttl_seconds: int = 3600

    def issue(
        self, *, user_id: str, team: str, tenant: str
    ) -> str:
        now = int(time.time())
        payload = {
            "sub": user_id,
            "team": team,
            "tenant": tenant,
            "iat": now,
            "exp": now + self.ttl_seconds,
        }
        return jwt.encode(payload, self.secret, ALGORITHM)

    def verify(self, token: str) -> Principal:
        try:
            claims = jwt.decode(
                token, self.secret, algorithms=[ALGORITHM]
            )
        except jwt.ExpiredSignatureError:
            raise AuthError("token expired") from None
        except jwt.InvalidTokenError:
            raise AuthError() from None
        missing = {"sub", "team", "tenant"} - set(claims)
        if missing:
            raise AuthError()
        return Principal(
            user_id=str(claims["sub"]),
            team=str(claims["team"]),
            clearance=str(claims["tenant"]),
        )


def bearer_token(
    authorization: str | None = Header(default=None),
) -> str:
    """Extract a bearer token, or refuse."""
    if not authorization:
        raise AuthError()
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthError()
    return token
