"""
Moderations 路由
"""

from fastapi import APIRouter, Depends, BackgroundTasks

from core.models import ModerationRequest
from routes.deps import rate_limit_dependency, verify_api_key, get_model_handler

router = APIRouter()


@router.post("/v1/moderations", dependencies=[Depends(rate_limit_dependency)])
async def moderations(
    request: ModerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    """
    内容审核
    
    兼容 OpenAI Moderations API 格式
    """
    model_handler = get_model_handler()
    return await model_handler.request_model(
        request, api_index, background_tasks, endpoint="/v1/moderations"
    )