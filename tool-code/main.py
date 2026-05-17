from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from admin_routes import router as admin_router
from app_config import BASE_DIR
from flowchart_routes import router as flowchart_router
from model_store import bootstrap_default_model
from user_routes import router as user_router
from vision_routes import router as vision_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_default_model()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.include_router(user_router)
app.include_router(vision_router)
app.include_router(flowchart_router)
app.include_router(admin_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
