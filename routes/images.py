"""
Images 路由
"""

from fastapi import APIRouter, Depends, BackgroundTasks

from core.models import ImageGenerationRequest
from routes.deps import rate_limit_dependency, verify_api_key, get_model_handler

router = APIRouter()


@router.post("/v1/images/generations", dependencies=[Depends(rate_limit_dependency)])
async def images_generations(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    """
    生成图像
    
    兼容 OpenAI Images API 格式
    """
    model_handler = get_model_handler()
    return await model_handler.request_model(
        request, api_index, background_tasks, endpoint="/v1/images/generations"
    )