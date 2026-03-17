"""
Streaming response helpers.

提供带统计和错误处理的流式响应包装器。
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import asyncio
from time import time

from starlette.responses import Response
from starlette.types import Scope, Receive, Send

from core.log_config import logger
from core.stats import update_stats
from core.utils import truncate_for_logging
from utils import safe_get


class LoggingStreamingResponse(Response):
    """
    包装底层流式响应：
    - 透传 chunk 给客户端
    - 解析 usage 字段，填充 current_info 中的 token 统计
    - 在完成后调用 update_stats 写入数据库
    """

    def __init__(
        self,
        content,
        status_code=200,
        headers=None,
        media_type=None,
        current_info=None,
        app=None,
        debug=False,
        dialect_id=None,
    ):
        super().__init__(content=None, status_code=status_code, headers=headers, media_type=media_type)
        self.body_iterator = content
        self._closed = False
        self.current_info = current_info or {}
        self.app = app
        self.debug = debug
        self.dialect_id = dialect_id or self.current_info.get("dialect_id")

        # Remove Content-Length header if it exists
        if "content-length" in self.headers:
            del self.headers["content-length"]
        # Set Transfer-Encoding to chunked
        self.headers["transfer-encoding"] = "chunked"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        try:
            async for chunk in self._logging_iterator():
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": True,
                    }
                )
        except Exception as e:
            # 记录异常但不重新抛出，避免"Task exception was never retrieved"
            logger.error(f"Error in streaming response: {type(e).__name__}: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()
            # 发送错误消息给客户端（如果可能）
            try:
                error_data = json.dumps({"error": f"Streaming error: {str(e)}"})
                await send(
                    {
                        "type": "http.response.body",
                        "body": f"data: {error_data}\n\n".encode("utf-8"),
                        "more_body": True,
                    }
                )
            except Exception as send_err:
                logger.error(f"Error sending error message: {str(send_err)}")
        finally:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )
            if hasattr(self.body_iterator, "aclose") and not self._closed:
                await self.body_iterator.aclose()
                self._closed = True

            # 记录处理时间并写入统计
            if "start_time" in self.current_info:
                process_time = time() - self.current_info["start_time"]
                self.current_info["process_time"] = process_time
            try:
                await update_stats(self.current_info, app=self.app)
            except Exception as e:
                logger.error(f"Error updating stats in LoggingStreamingResponse: {str(e)}")

    def _split_complete_stream_lines(self, buffer: str, chunk_text: str) -> Tuple[List[str], str]:
        """
        将流式文本切分为完整行，并保留跨 chunk 的残留数据。

        说明：
        - 仅输出已遇到换行符的完整行
        - 未结束的最后一行保留到下一次 chunk 再解析
        - 统一兼容 \n / \r\n
        """
        combined = buffer + chunk_text
        if not combined:
            return [], ""

        lines: List[str] = []
        start = 0

        for idx, char in enumerate(combined):
            if char == "\n":
                line = combined[start:idx]
                if line.endswith("\r"):
                    line = line[:-1]
                lines.append(line)
                start = idx + 1

        return lines, combined[start:]

    def _extract_sse_payload(self, line: str) -> Optional[str]:
        """从单行 SSE 数据中提取 JSON payload。"""
        stripped = line.strip()

        # 跳过空行、注释行、事件名行
        if not stripped or stripped.startswith(":") or stripped.startswith("event:"):
            return None

        if stripped.startswith("data:"):
            stripped = stripped[5:].strip()

        if not stripped or stripped.startswith("[DONE]") or stripped.startswith("OK"):
            return None

        return stripped

    def _extract_usage_info(self, resp: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """按当前方言优先、OpenAI 保底的顺序提取 usage。"""
        from core.dialects.registry import get_dialect

        d_id = self.dialect_id or self.current_info.get("dialect_id") or "openai"
        dialect = get_dialect(d_id)

        usage_info = None
        if dialect and dialect.parse_usage:
            usage_info = dialect.parse_usage(resp)

        # 某些链路会先转换为 Canonical(OpenAI 风格)，这里用 openai 兜底
        if not usage_info and d_id != "openai":
            o_dialect = get_dialect("openai")
            if o_dialect and o_dialect.parse_usage:
                usage_info = o_dialect.parse_usage(resp)

        return usage_info

    def _merge_usage_info(self, usage_info: Optional[Dict[str, int]]) -> None:
        """
        以“非零覆盖 + 补全”的方式合并 usage，避免分阶段上报时互相覆盖。
        """
        if not usage_info:
            return

        old_prompt = int(self.current_info.get("prompt_tokens") or 0)
        old_completion = int(self.current_info.get("completion_tokens") or 0)
        old_total = int(self.current_info.get("total_tokens") or 0)

        new_prompt = int(usage_info.get("prompt_tokens") or 0)
        new_completion = int(usage_info.get("completion_tokens") or 0)
        new_total = int(usage_info.get("total_tokens") or 0)

        prompt_tokens = new_prompt if new_prompt > 0 else old_prompt
        completion_tokens = new_completion if new_completion > 0 else old_completion

        # total_tokens 优先使用新值；若新值缺失或偏小，则用已知 prompt/completion 补全
        total_tokens = new_total if new_total > 0 else old_total
        derived_total = prompt_tokens + completion_tokens
        if derived_total > 0 and derived_total > total_tokens:
            total_tokens = derived_total

        self.current_info["prompt_tokens"] = prompt_tokens
        self.current_info["completion_tokens"] = completion_tokens
        self.current_info["total_tokens"] = total_tokens

    async def _process_stream_line(self, line: str, content_start_recorded: bool) -> bool:
        """解析单行 SSE/JSON 数据，更新正文开始时间与 usage 统计。"""
        payload = self._extract_sse_payload(line)
        if not payload:
            return content_start_recorded

        try:
            resp = await asyncio.to_thread(json.loads, payload)

            if not content_start_recorded:
                choices = resp.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    content = safe_get(choices[0], "delta", "content", default=None)
                    if content and content.strip():
                        self.current_info["content_start_time"] = time() - self.current_info.get("start_time", time())
                        content_start_recorded = True

            usage_info = self._extract_usage_info(resp)
            self._merge_usage_info(usage_info)
        except Exception as e:
            if self.debug:
                logger.error(f"Error parsing streaming response: {str(e)}, line: {repr(payload)}")

        return content_start_recorded

    async def _logging_iterator(self):
        # 用于收集响应体的缓冲区（仅在配置了保留时间时使用）
        # response_chunks 用于收集返回给用户的响应（即经过转换后的）
        response_chunks = []
        max_response_size = 100 * 1024  # 100KB
        total_response_size = 0
        should_save_response = self.current_info.get("raw_data_expires_at") is not None
        content_start_recorded = False  # 标记是否已记录正文开始时间
        line_buffer = ""
        
        async for chunk in self.body_iterator:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")

            # 收集响应体（限制大小）
            if should_save_response and total_response_size < max_response_size:
                response_chunks.append(chunk)
                total_response_size += len(chunk)

            # 音频流不解析 usage，直接透传
            if self.current_info.get("endpoint", "").endswith("/v1/audio/speech"):
                yield chunk
                continue

            # 使用 errors="replace" 避免解码错误导致流终止
            chunk_text = chunk.decode("utf-8", errors="replace")
            if self.debug:
                logger.info(chunk_text.encode("utf-8").decode("unicode_escape"))

            # 使用行缓冲区处理跨 chunk 拆分的 SSE 数据
            lines, line_buffer = self._split_complete_stream_lines(line_buffer, chunk_text)
            for line in lines:
                content_start_recorded = await self._process_stream_line(line, content_start_recorded)
            
            # 透传原始 chunk
            yield chunk

        if line_buffer:
            content_start_recorded = await self._process_stream_line(line_buffer, content_start_recorded)
        
        # 保存返回给用户的响应体（使用深度截断，保留结构同时限制大小）
        # 使用 asyncio.to_thread 避免大响应体阻塞事件循环
        if should_save_response and response_chunks:
            try:
                response_body = b"".join(response_chunks)
                self.current_info["response_body"] = await asyncio.to_thread(truncate_for_logging, response_body)
            except Exception as e:
                logger.error(f"Error saving response body: {str(e)}")

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if hasattr(self.body_iterator, "aclose"):
                await self.body_iterator.aclose()