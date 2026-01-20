import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.meal import router as meal_router
from app.api.chat import router as chat_router

app = FastAPI(
    title="属于韩晗的一周食谱",
    description="根据一周天气与当季蔬菜价格推荐一周食谱",
    version="0.1.0",
)

# CORS 配置 (允许前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(meal_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "name": "FoodPlanGPT API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/api/chat",
            "meal": "/api/meal",
            "context": "/api/chat/context",
            "health": "/api/chat/health"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)