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
from typing import Any, Optional, Union


def normalize_error_detail(detail: Any) -> tuple[str, Optional[Union[str, int]], Optional[str], Any]:
    """将异常 detail 归一化为 message/code/param/raw_detail。"""
    if isinstance(detail, dict):
        message = (
            detail.get("message")
            or detail.get("detail")
            or detail.get("error")
            or str(detail)
        )
        code = detail.get("error_code") or detail.get("code")
        param = detail.get("param")
        return str(message), code, param, detail

    if detail is None:
        return "", None, None, None

    return str(detail), None, None, detail


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


DEFAULT_PROXY_ERROR_MESSAGE = "Reverse proxy request failed."
DEFAULT_PROXY_ERROR_CODE = "reverse_proxy_error"
DEFAULT_PROXY_ERROR_TYPE = "api_error"
DEFAULT_PROXY_ERROR_STATUS = 502


def normalize_proxy_error_policy(policy: Any) -> Optional[dict[str, Any]]:
    """将插件/运行时配置归一化为统一反代错误策略。"""
    if policy in (None, False, "", {}):
        return None

    if policy is True:
        policy = {}
    elif isinstance(policy, str):
        policy = {"message": policy}
    elif not isinstance(policy, dict):
        policy = {}

    status_code = policy.get("status_code", DEFAULT_PROXY_ERROR_STATUS)
    try:
        status_code = int(status_code)
    except Exception:
        status_code = DEFAULT_PROXY_ERROR_STATUS
    if status_code < 100 or status_code > 599:
        status_code = DEFAULT_PROXY_ERROR_STATUS

    message = str(policy.get("message") or DEFAULT_PROXY_ERROR_MESSAGE)
    error_type = str(policy.get("error_type") or DEFAULT_PROXY_ERROR_TYPE)
    code = policy.get("code", DEFAULT_PROXY_ERROR_CODE)
    if code is not None:
        code = str(code)

    return {
        "status_code": status_code,
        "message": message,
        "error_type": error_type,
        "code": code,
    }


def create_error_response(
    message: str,
    status_code: int = 500,
    error_type: Optional[str] = None,
    param: Optional[str] = None,
    code: Optional[Union[str, int]] = None,
    detail: Optional[Any] = None,
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
    
    response_content: dict[str, Any] = {
        "error": error_content,
        "detail": message,
    }
    if code is not None:
        response_content["error_code"] = code
    if isinstance(detail, dict):
        response_content["details"] = detail

    return JSONResponse(
        status_code=status_code,
        content=response_content
    )


def openai_error_response(
    message: str,
    status_code: int = 500,
    *,
    code: Optional[Union[str, int]] = None,
    detail: Optional[Any] = None,
) -> JSONResponse:
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
    return create_error_response(message=message, status_code=status_code, code=code, detail=detail)


def build_proxy_error_dict(policy: Any) -> dict[str, Any]:
    normalized = normalize_proxy_error_policy(policy) or normalize_proxy_error_policy(True)
    return {
        "error": {
            "message": normalized["message"],
            "type": normalized["error_type"],
            "code": normalized["code"],
        },
        "status_code": normalized["status_code"],
        "details": {
            "message": normalized["message"],
            "code": normalized["code"],
            "proxy_error": True,
        },
    }


def create_proxy_error_response(policy: Any) -> JSONResponse:
    normalized = normalize_proxy_error_policy(policy) or normalize_proxy_error_policy(True)
    return create_error_response(
        message=normalized["message"],
        status_code=normalized["status_code"],
        error_type=normalized["error_type"],
        code=normalized["code"],
        detail=None,
    )
