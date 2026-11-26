from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model import get_artifacts
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_artifacts()
    yield


app = FastAPI(title="Medical Cost Prediction API", version="1.0.0", lifespan=lifespan)
app.include_router(router)
