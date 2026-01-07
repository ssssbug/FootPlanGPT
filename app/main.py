import uvicorn
from fastapi import FastAPI
from app.api.meal import router as meal_router
app = FastAPI(
    title="属于韩晗的一周食谱",
    description="根据一周天气与当季蔬菜价格推荐一周食谱",
    version="0.1.0",
)
app.include_router(meal_router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)