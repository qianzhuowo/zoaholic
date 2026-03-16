# 多模态标准 Schema（Zoaholic）

本文档描述 Zoaholic 当前对多模态输入（文本 / 图片 / 文件）的统一约定，方便前端、插件、渠道适配器和第三方调用方对齐。

## 1. 网关入口消息 Schema

### 1.1 Message

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "请阅读附件并总结" },
    {
      "type": "file",
      "file": {
        "url": "data:text/plain;base64,SGVsbG8=",
        "mime_type": "text/plain",
        "filename": "note.txt"
      }
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://example.com/cat.png"
      }
    }
  ]
}
```

其中 `content` 支持三类 item：

- `text`
- `image_url`
- `file`

### 1.2 FileRef

网关标准 `FileRef` 字段如下：

```json
{
  "url": "data:... 或 https://...",
  "data": "data:...（url 的别名）",
  "mime_type": "application/pdf",
  "mimeType": "application/pdf",
  "filename": "report.pdf",
  "file_name": "report.pdf",
  "name": "report.pdf",
  "file_id": "file-xxx",
  "fileId": "file-xxx"
}
```

说明：

- `url` / `data`：至少提供一个；当前支持 `data:` 和 `http(s)://`。
- `mime_type` / `mimeType`：可选；缺省时会按远端响应头或文件名推断。
- `filename` / `file_name` / `name`：文件展示名。
- `file_id` / `fileId`：供支持文件引用的渠道使用，例如 OpenAI Responses API。

## 2. 统一文件处理工具

统一文件处理入口位于：

- `core/file_utils.py`
  - `resolve_file_ref(...)`
  - `normalize_file_ref(...)`
  - `FileRefProcessor`
  - `NormalizedFileRef`

### 2.1 ResolvedFileRef

用于表示已经拿到真实字节内容的附件：

```json
{
  "filename": "note.txt",
  "mime_type": "text/plain",
  "data_url": "data:text/plain;base64,SGVsbG8=",
  "content": "<bytes>"
}
```

适用场景：

- 渠道需要真实二进制内容
- 需要转成 base64 / inlineData
- 需要提取文本内容

### 2.2 NormalizedFileRef（内部标准 Schema）

`normalize_file_ref(...)` 会把 `file_id` 引用与真实文件统一成一套内部结构：

```json
{
  "schema_version": "multimodal.file.v1",
  "type": "file",
  "source": "data_url | http_url | file_id",
  "filename": "note.txt",
  "mime_type": "text/plain",
  "size_bytes": 5,
  "file_id": null,
  "source_url": null,
  "capabilities": {
    "resolved": true,
    "image": false,
    "pdf": false,
    "text": true
  },
  "text_content": "Hello"
}
```

字段含义：

- `schema_version`：内部标准版本，便于后续演进。
- `source`：附件来源。
  - `data_url`
  - `http_url`
  - `file_id`
- `size_bytes`：真实内容大小；`file_id`-only 场景通常为 `0`。
- `capabilities.resolved`：是否已经拿到真实文件数据。
- `capabilities.image/pdf/text`：基于 MIME 判断的能力标记。
- `text_content`：仅文本类附件会有值。

### 2.3 FileRefProcessor

`FileRefProcessor` 是对函数式工具的轻量封装，适合在渠道适配器/插件中直接复用：

- `await FileRefProcessor.resolve(file_ref)`
- `await FileRefProcessor.normalize(file_ref)`
- `FileRefProcessor.to_schema(normalized)`
- `FileRefProcessor.render_text_attachment(normalized)`

## 3. 渠道侧映射约定

### 3.1 OpenAI Chat Completions

- 图片文件 → `image_url`
- 纯文本文件 → 转成 `text`
- `file_id` / PDF / 二进制文件 → 不直接支持，建议改用 Responses API

### 3.2 OpenAI Responses

- 图片文件 → `input_image`
- 其他文件 → `input_file`
- `file_id` → 直接透传为 `input_file.file_id`
- 纯文本文件在部分场景下可降级为 `input_text`，提升兼容性

### 3.3 Claude

- 图片 → `image`
- PDF → `document`
- 纯文本 → `text`

### 3.4 Gemini

- 图片 / PDF / 其他二进制 → `inlineData`
- `file_id` 当前不支持，需要传真实文件内容

## 4. 错误约定

统一由 `core/file_utils.py` 返回 `FileRef 校验失败：...` 风格错误，例如：

- `FileRef 校验失败：缺少 file.url / file.data`
- `FileRef 校验失败：附件内容必须是合法的 data URL`
- `FileRef 校验失败：OpenAI Chat Completions 暂不支持附件类型 application/pdf。支持范围：图片、纯文本附件；若需 PDF/二进制文件请改用 Responses API、Claude 或 Gemini`

## 5. 前端 / 第三方调用示例合集

下面给出几组可直接参考的调用方式，统一假设：

- 网关地址：`https://your-gateway.example.com`
- 网关 Key：`sk-your-gateway-key`

### 5.1 浏览器前端：把本地文本文件转成 data URL 后发送

适用场景：

- 前端上传 `.txt` / `.md` / `.json` / `.csv`
- 希望直接走 `/v1/chat/completions`
- 附件本质会被网关转成文本内容注入模型输入

```ts
async function fileToDataUrl(file: File): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function sendTextAttachment(file: File) {
  const dataUrl = await fileToDataUrl(file);

  const body = {
    model: "gpt-4o-mini",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "请阅读附件并总结重点" },
          {
            type: "file",
            file: {
              url: dataUrl,
              mime_type: file.type || "text/plain",
              filename: file.name,
            },
          },
        ],
      },
    ],
  };

  const resp = await fetch("https://your-gateway.example.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer sk-your-gateway-key",
    },
    body: JSON.stringify(body),
  });

  return await resp.json();
}
```

### 5.2 浏览器前端：图片 URL + 文件混合输入

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "请结合图片和附件一起分析" },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/demo-chart.png"
          }
        },
        {
          "type": "file",
          "file": {
            "url": "data:text/markdown;base64,IyDmtYvor5UKCi0gQTEKLSBCMgo=",
            "mime_type": "text/markdown",
            "filename": "report.md"
          }
        }
      ]
    }
  ]
}
```

说明：

- `image_url` 适合图片引用
- `file` 适合文本/PDF/二进制文件输入
- 如果目标是 OpenAI Chat Completions 路径，建议文件类型优先使用纯文本或图片

### 5.3 前端 / 服务端：直接引用远程文件 URL

如果附件已经在公网可访问，可以直接传远程 URL，不必先转 base64：

```json
{
  "type": "file",
  "file": {
    "url": "https://example.com/files/meeting-notes.txt",
    "mime_type": "text/plain",
    "filename": "meeting-notes.txt"
  }
}
```

适合：

- OSS / S3 / CDN 上已存在的文件
- 服务端中转时不想额外做 base64 编码

注意：

- URL 必须能被网关服务端访问到
- 若不传 `mime_type`，网关会尝试从响应头或文件名推断

### 5.4 第三方 Python（requests）：发送文本附件到 `/v1/chat/completions`

```python
import base64
import requests

gateway_base = "https://your-gateway.example.com"
gateway_key = "sk-your-gateway-key"

raw = open("note.txt", "rb").read()
data_url = "data:text/plain;base64," + base64.b64encode(raw).decode("utf-8")

payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请阅读附件后输出 3 条摘要"},
                {
                    "type": "file",
                    "file": {
                        "url": data_url,
                        "mime_type": "text/plain",
                        "filename": "note.txt",
                    },
                },
            ],
        }
    ],
}

resp = requests.post(
    f"{gateway_base}/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {gateway_key}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=120,
)

print(resp.status_code)
print(resp.json())
```

### 5.5 第三方 Python（requests）：发送 PDF / 二进制文件到 `/v1/responses`

当附件是 PDF、Office 文档、压缩包等非纯文本文件时，建议走 Responses 风格入口：

```python
import base64
import requests

raw = open("report.pdf", "rb").read()
data_url = "data:application/pdf;base64," + base64.b64encode(raw).decode("utf-8")

payload = {
    "model": "gpt-4.1",
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "请阅读这份 PDF 并输出摘要"},
                {
                    "type": "file",
                    "file": {
                        "url": data_url,
                        "mime_type": "application/pdf",
                        "filename": "report.pdf",
                    },
                },
            ],
        }
    ],
}
```

> 说明：Zoaholic 会在多方言转换阶段把 `type=file` 统一映射到目标渠道所需格式。

### 5.6 走 `file_id` 引用（适合 OpenAI Responses 风格场景）

如果你的上游或业务侧已经有可复用的文件引用 ID，可以这样传：

```json
{
  "model": "gpt-4.1",
  "input": [
    {
      "role": "user",
      "content": [
        { "type": "input_text", "text": "请读取这个已上传文件并总结" },
        {
          "type": "file",
          "file": {
            "file_id": "file-abc123",
            "filename": "report.pdf"
          }
        }
      ]
    }
  ]
}
```

说明：

- `file_id` 主要适合支持文件引用的渠道
- 当前 Gemini 这类需要真实文件内容的渠道，不适合只传 `file_id`

### 5.7 cURL：远程 URL 文件示例

`curl` 直接传大段 base64 不太方便，更推荐演示远程 URL：

```bash
curl https://your-gateway.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "请读取这个远程文本文件并总结"},
          {
            "type": "file",
            "file": {
              "url": "https://example.com/files/readme.txt",
              "mime_type": "text/plain",
              "filename": "readme.txt"
            }
          }
        ]
      }
    ]
  }'
```

### 5.8 推荐的前端封装格式

如果你在前端自己维护附件列表，建议统一整理成下面这种结构后再发请求：

```ts
type GatewayContentItem =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | {
      type: "file";
      file: {
        url?: string;
        data?: string;
        mime_type?: string;
        filename?: string;
        file_id?: string;
      };
    };
```

这样可以做到：

- 本地文件 → 转 `data:`
- 远程文件 → 保留 `url`
- 引用型文件 → 使用 `file_id`
- 前端状态管理与网关协议一一对应

### 5.9 按方言拆分的示例

如果你的调用方不是统一走 OpenAI 兼容入口，而是希望直接按方言调用 Zoaholic，可以参考下面几组示例。

#### 5.9.1 OpenAI Chat Completions 方言：`POST /v1/chat/completions`

这是当前 **最推荐的通用多模态入口**，也是 Zoaholic 内部的 Canonical 形式。

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "请结合图片和文本附件一起总结" },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/chart.png"
          }
        },
        {
          "type": "file",
          "file": {
            "url": "data:text/plain;base64,SGVsbG8sIFpvYWhvbGljIQ==",
            "mime_type": "text/plain",
            "filename": "note.txt"
          }
        }
      ]
    }
  ],
  "stream": false
}
```

适合：

- 文本 + 图片 + 文件混合输入
- `data URL` / 远程 URL / `file_id` 的统一接入
- 前端 Playground、自建前端、第三方服务端 SDK 接入

#### 5.9.2 OpenAI Responses 方言：`POST /v1/responses`

适合 PDF、通用文件、`file_id` 引用等更偏“文件输入”风格的场景。

```json
{
  "model": "gpt-4.1",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "请阅读这份 PDF 并输出摘要"
        },
        {
          "type": "file",
          "file": {
            "url": "data:application/pdf;base64,JVBERi0xLjcKJc...",
            "mime_type": "application/pdf",
            "filename": "report.pdf"
          }
        }
      ]
    }
  ]
}
```

也可以直接传文件引用：

```json
{
  "model": "gpt-4.1",
  "input": [
    {
      "role": "user",
      "content": [
        { "type": "input_text", "text": "请读取已上传文件并总结" },
        {
          "type": "file",
          "file": {
            "file_id": "file-abc123",
            "filename": "report.pdf"
          }
        }
      ]
    }
  ]
}
```

#### 5.9.3 Claude 方言：`POST /v1/messages`

Claude 原生入口建议优先使用 **文本 + 图片 block**。如果你要传 PDF/通用文件，当前更推荐改走 OpenAI 兼容入口或 Responses 入口。

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "system": "你是一个善于识图和总结的助手。",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "请描述图片并总结重点" },
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgoAAAANSUhEUgAA..."
          }
        }
      ]
    }
  ]
}
```

说明：

- Claude 方言入口最稳妥的多模态形式是 `text` + `image`
- 若你的前端统一维护 `type=file` 结构，建议直接走 `/v1/chat/completions` 让网关统一转换

#### 5.9.4 Gemini 方言：`POST /v1beta/models/{model}:generateContent`

Gemini 原生入口建议使用 `contents[].parts[]`，其中图片/二进制内容通过 `inlineData` 传入。

```json
{
  "systemInstruction": {
    "parts": [
      { "text": "你是一个会阅读图片内容的助手。" }
    ]
  },
  "contents": [
    {
      "role": "user",
      "parts": [
        { "text": "请识别图片中的图表信息" },
        {
          "inlineData": {
            "mimeType": "image/png",
            "data": "iVBORw0KGgoAAAANSUhEUgAA..."
          }
        }
      ]
    }
  ],
  "generationConfig": {
    "temperature": 0.2
  }
}
```

Gemini 鉴权可使用：

- 请求头：`x-goog-api-key: sk-your-gateway-key`
- 或查询参数：`?key=sk-your-gateway-key`

说明：

- Gemini 方言入口当前最适合 `text + inlineData`
- 若你有远程 URL、`file_id`、通用 `type=file` 结构，推荐仍优先走 OpenAI 兼容入口，由 Zoaholic 统一处理

## 6. 建议实践

1. **前端尽量补齐 `mime_type` 与 `filename`**，减少推断误差。
2. **大文本附件要做长度控制**，网关会在转文本时自动截断。
3. **如果目标渠道支持 `file_id`，优先走引用**，避免把大文件重复转成 data URL。
4. **插件如果需要处理附件，请优先使用 `normalize_file_ref` / `FileRefProcessor`**，不要在插件里重复手写 `data_url.split(",")`、MIME 判断和文本解码逻辑。
