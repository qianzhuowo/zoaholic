"""
错误响应处理模块

提供标准化的 OpenAI 风格错误响应，确保所有错误返回格式一致。

错误类型说明：
- invalid_request_error: 请求参数错误 (400, 413, 422)
- authentication_error: 认证失败 (401)
- permission_error/permission_denied_error: 权限不足 (403)
- not_found_error: 资源不存在 (404)
- rate_limit_error/rate_limit_exceeded: 请求频率限制 (429)
- api_error/internal_server_error: 服务器内部错误 (500+)
- service_unavailable_error: 服务不可用 (503)
"""

from fastapi.responses import JSONResponse
from typing import Optional, Union


# OpenAI 标准错误类型映射
ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_denied_error",
    404: "not_found_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_exceeded",
    500: "internal_server_error",
    502: "api_error",
    503: "service_unavailable_error",
    504: "api_error",
}


def create_error_response(
    message: str,
    status_code: int = 500,
    error_type: Optional[str] = None,
    param: Optional[str] = None,
    code: Optional[Union[str, int]] = None
) -> JSONResponse:
    """
    创建标准 OpenAI 风格的错误响应
    
    参数:
        message: 错误信息描述
        status_code: HTTP 状态码
        error_type: 错误类型，如果为 None 则根据 status_code 自动推断
        param: 触发错误的参数名（可选）
        code: 错误代码（可选）
    
    返回:
        JSONResponse: 标准化的错误响应
    
    示例响应格式:
    {
        "error": {
            "message": "Invalid API Key",
            "type": "authentication_error",
            "param": null,
            "code": null
        }
    }
    """
    # 如果未指定 error_type，根据状态码自动推断
    if error_type is None:
        error_type = ERROR_TYPE_MAP.get(status_code, "api_error")
    
    error_content = {
        "message": message,
        "type": error_type,
    }
    
    # 只在有值时添加 param 和 code 字段
    if param is not None:
        error_content["param"] = param
    if code is not None:
        error_content["code"] = code
    
    return JSONResponse(
        status_code=status_code,
        content={"error": error_content}
    )


def openai_error_response(message: str, status_code: int = 500) -> JSONResponse:
    """
    便捷方法：创建 OpenAI 风格错误响应
    
    自动根据状态码推断错误类型，简化调用。
    
    参数:
        message: 错误信息
        status_code: HTTP 状态码
    
    返回:
        JSONResponse: 标准化的错误响应
    
    示例:
        >>> response = openai_error_response("Invalid API Key", 403)
        >>> # 返回 403 响应，type="permission_denied_error"
    """
    return create_error_response(message=message, status_code=status_code)
