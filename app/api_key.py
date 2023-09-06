from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyQuery, APIKeyCookie, APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from app.config import API_KEY_NAME, API_KEY

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_cookie = APIKeyCookie(name=API_KEY_NAME, auto_error=False)


async def get_api_key(
    query: str = Security(api_key_query),
    header: str = Security(api_key_header),
    cookie: str = Security(api_key_cookie),
):

    if query == API_KEY:
        return query
    elif header == API_KEY:
        return header
    elif cookie == API_KEY:
        return cookie
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )
