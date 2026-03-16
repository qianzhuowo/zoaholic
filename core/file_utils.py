"""
统一文件/附件处理工具。

当前阶段聚焦于：
- FileRef 统一解析
- data URL / 远程 URL 转标准化字节流
- MIME 判断
- 文本类附件提取
- 对渠道/插件暴露统一的标准化文件对象
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from typing import Any, Optional, Literal
from urllib.parse import unquote, urlparse

import httpx
from fastapi import HTTPException

from core.log_config import logger


def fileref_error_message(message: str) -> str:
    return f"FileRef 校验失败：{message}"


def raise_fileref_error(message: str, status_code: int = 400) -> None:
    raise HTTPException(status_code=status_code, detail=fileref_error_message(message))


def raise_fileref_unsupported_error(channel: str, mime_type: Optional[str], supported_hint: str) -> None:
    target_mime = mime_type or "unknown"
    raise_fileref_error(f"{channel} 暂不支持附件类型 {target_mime}。支持范围：{supported_hint}")


TEXT_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/javascript",
    "application/x-javascript",
    "text/csv",
    "text/markdown",
}


@dataclass
class ResolvedFileRef:
    filename: Optional[str]
    mime_type: str
    data_url: str
    content: bytes


@dataclass
class NormalizedFileRef:
    """统一后的文件引用结构。

    - source=file_id：仅有文件引用，还未解析出真实文件内容
    - source=data_url/http_url：已解析出真实内容，可供渠道进一步转换
    """

    source: Literal["file_id", "data_url", "http_url"]
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    file_id: Optional[str] = None
    data_url: Optional[str] = None
    base64_data: Optional[str] = None
    content: Optional[bytes] = None
    text_content: Optional[str] = None
    size_bytes: int = 0
    source_url: Optional[str] = None

    @property
    def is_resolved(self) -> bool:
        return self.data_url is not None and self.content is not None

    @property
    def is_image(self) -> bool:
        return is_image_mime_type(self.mime_type)

    @property
    def is_pdf(self) -> bool:
        return is_pdf_mime_type(self.mime_type)

    @property
    def is_text(self) -> bool:
        return self.text_content is not None

    def render_text_attachment(self, *, max_chars: int = 40000) -> Optional[str]:
        if self.text_content is None:
            return None
        return render_text_attachment_content(
            self.filename,
            self.mime_type or "application/octet-stream",
            self.text_content,
            max_chars=max_chars,
        )

    def to_schema(self, *, include_data_url: bool = False, include_text_content: bool = True) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "schema_version": "multimodal.file.v1",
            "type": "file",
            "source": self.source,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "file_id": self.file_id,
            "source_url": self.source_url,
            "capabilities": {
                "resolved": self.is_resolved,
                "image": self.is_image,
                "pdf": self.is_pdf,
                "text": self.is_text,
            },
        }
        if include_text_content and self.text_content is not None:
            schema["text_content"] = self.text_content
        if include_data_url and self.data_url is not None:
            schema["data_url"] = self.data_url
        return schema


class FileRefProcessor:
    """统一文件处理工具类，供渠道适配器和插件复用。"""

    @staticmethod
    async def resolve(file_ref) -> ResolvedFileRef:
        return await resolve_file_ref(file_ref)

    @staticmethod
    async def normalize(file_ref) -> NormalizedFileRef:
        return await normalize_file_ref(file_ref)

    @staticmethod
    def extract_text(resolved: ResolvedFileRef) -> Optional[str]:
        return extract_text_from_resolved_file(resolved)

    @staticmethod
    def render_text_attachment(normalized: NormalizedFileRef, *, max_chars: int = 40000) -> Optional[str]:
        return normalized.render_text_attachment(max_chars=max_chars)

    @staticmethod
    def to_schema(normalized: NormalizedFileRef, *, include_data_url: bool = False, include_text_content: bool = True) -> dict[str, Any]:
        return normalized.to_schema(include_data_url=include_data_url, include_text_content=include_text_content)


async def fetch_file_from_url(url: str) -> tuple[bytes, Optional[str]]:
    transport = httpx.AsyncHTTPTransport(
        http2=True,
        verify=False,
        retries=1,
    )
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            content_type = response.headers.get("content-type")
            if content_type:
                content_type = content_type.split(";", 1)[0].strip() or None
            return response.content, content_type
        except httpx.RequestError as e:
            logger.error(f"请求 URL 时出错 {e.request.url!r}: {e}")
            raise_fileref_error(f"无法从 URL 获取附件内容：{url}")
        except httpx.HTTPStatusError as e:
            logger.error(f"获取 URL 时发生 HTTP 错误 {e.request.url!r}: {e.response.status_code}")
            raise_fileref_error(f"获取附件 URL 失败（HTTP {e.response.status_code}）：{url}", status_code=e.response.status_code)


def guess_mime_type(filename: Optional[str], default: str = "application/octet-stream") -> str:
    if not filename:
        return default
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or default


def infer_filename_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path or "")
        if not path:
            return None
        name = path.rsplit("/", 1)[-1].strip()
        return name or None
    except Exception:
        return None


def is_image_mime_type(mime_type: Optional[str]) -> bool:
    return bool(mime_type and str(mime_type).lower().startswith("image/"))


def is_pdf_mime_type(mime_type: Optional[str]) -> bool:
    return str(mime_type or "").lower() == "application/pdf"


def is_text_mime_type(mime_type: Optional[str]) -> bool:
    normalized = str(mime_type or "").lower()
    return normalized.startswith("text/") or normalized in TEXT_MIME_TYPES


def build_data_url(content: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def split_data_url(data_url: str) -> tuple[str, bytes]:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise_fileref_error("附件内容必须是合法的 data URL")

    try:
        header, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=fileref_error_message("附件 data URL 格式不正确")) from exc

    meta = header[5:]
    parts = [part.strip() for part in meta.split(";") if part.strip()]
    mime_type = parts[0] if parts else "application/octet-stream"
    is_base64 = any(part.lower() == "base64" for part in parts[1:])
    if not is_base64:
        raise_fileref_error("当前仅支持 base64 编码的 data URL 附件")

    try:
        return mime_type, base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=fileref_error_message("附件 base64 数据无效")) from exc


def extract_data_url_base64(data_url: str) -> str:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise_fileref_error("附件内容必须是合法的 data URL")
    try:
        _header, encoded = data_url.split(",", 1)
        return encoded
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=fileref_error_message("附件 data URL 格式不正确")) from exc


async def resolve_file_ref(file_ref) -> ResolvedFileRef:
    if file_ref is None:
        raise_fileref_error("缺少 file 引用对象")

    source = getattr(file_ref, "url", None) or getattr(file_ref, "data", None)
    filename = getattr(file_ref, "filename", None) or getattr(file_ref, "name", None)
    mime_type = getattr(file_ref, "mime_type", None) or getattr(file_ref, "mimeType", None)

    if not source:
        raise_fileref_error("缺少 file.url / file.data")

    if isinstance(source, str) and source.startswith("data:"):
        parsed_mime_type, content = split_data_url(source)
        final_mime_type = mime_type or parsed_mime_type
        return ResolvedFileRef(
            filename=filename,
            mime_type=final_mime_type,
            data_url=source if final_mime_type == parsed_mime_type else build_data_url(content, final_mime_type),
            content=content,
        )

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        content, remote_mime_type = await fetch_file_from_url(source)
        inferred_filename = filename or infer_filename_from_url(source)
        final_mime_type = mime_type or remote_mime_type or guess_mime_type(inferred_filename)
        return ResolvedFileRef(
            filename=inferred_filename,
            mime_type=final_mime_type,
            data_url=build_data_url(content, final_mime_type),
            content=content,
        )

    raise_fileref_error("file.url 仅支持 data URL 或 http(s) URL")


async def normalize_file_ref(file_ref) -> NormalizedFileRef:
    if file_ref is None:
        raise_fileref_error("缺少 file 引用对象")

    file_id = getattr(file_ref, "file_id", None) or getattr(file_ref, "fileId", None)
    filename = getattr(file_ref, "filename", None) or getattr(file_ref, "name", None)
    mime_type = getattr(file_ref, "mime_type", None) or getattr(file_ref, "mimeType", None)
    source = getattr(file_ref, "url", None) or getattr(file_ref, "data", None)

    if file_id and not source:
        return NormalizedFileRef(
            source="file_id",
            filename=filename,
            mime_type=mime_type,
            file_id=file_id,
        )

    resolved = await resolve_file_ref(file_ref)
    text_content = extract_text_from_resolved_file(resolved)
    source_kind: Literal["data_url", "http_url"] = "data_url"
    source_url = None
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        source_kind = "http_url"
        source_url = source

    return NormalizedFileRef(
        source=source_kind,
        filename=resolved.filename,
        mime_type=resolved.mime_type,
        file_id=file_id,
        data_url=resolved.data_url,
        base64_data=extract_data_url_base64(resolved.data_url),
        content=resolved.content,
        text_content=text_content,
        size_bytes=len(resolved.content),
        source_url=source_url,
    )


def require_resolved_file_data(normalized: NormalizedFileRef, channel: str, supported_hint: str) -> tuple[str, str]:
    if not normalized.is_resolved or not normalized.base64_data:
        raise_fileref_unsupported_error(channel, normalized.mime_type, supported_hint)
    return normalized.mime_type or "application/octet-stream", normalized.base64_data


def require_text_file_content(normalized: NormalizedFileRef, channel: str, supported_hint: str) -> str:
    if normalized.text_content is None:
        raise_fileref_unsupported_error(channel, normalized.mime_type, supported_hint)
    return normalized.text_content


def extract_text_from_resolved_file(resolved: ResolvedFileRef) -> Optional[str]:
    if not is_text_mime_type(resolved.mime_type):
        return None

    raw = resolved.content
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    try:
        return raw.decode("latin-1")
    except Exception:
        return None


def render_text_attachment_content(
    filename: Optional[str],
    mime_type: str,
    text: str,
    *,
    max_chars: int = 40000,
) -> str:
    normalized_text = str(text or "")
    if len(normalized_text) > max_chars:
        normalized_text = normalized_text[:max_chars] + f"\n\n[...附件内容已截断，原始长度约 {len(text)} 字符]"
    display_name = filename or "attachment"
    return f"[附件: {display_name} ({mime_type})]\n{normalized_text}"
