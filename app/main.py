import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from app.config import SENTRY_DSN, ENVIRONMENT, SENTRY_TRACE_SAMPLE_RATE
from app.routers import load_routers


def traces_sampler(sampling_context):
    asgi_scope = sampling_context["asgi_scope"]
    if asgi_scope["method"] == "GET" and asgi_scope["path"] == "/health":  # Exclude health_check
        return 0
    else:
        # Default sample rate
        return SENTRY_TRACE_SAMPLE_RATE


sentry_sdk.init(dsn=SENTRY_DSN,
                environment=ENVIRONMENT,
                traces_sampler=traces_sampler
                )


def get_app():
    create_app = FastAPI()

    origins = [
        "http://localhost:3000",
        "http://localhost:8080",
    ]

    create_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    load_routers(create_app)
    return create_app


# init elastic server and model


fast_api_app = get_app()
app = SentryAsgiMiddleware(fast_api_app)
