from fastapi import FastAPI

from backend import routes

app = FastAPI()

app = FastAPI(title="Face Verification API")

app.include_router(routes.router, prefix="/api/v1")
