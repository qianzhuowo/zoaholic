# Docker CA 信任库验收

生产 `Dockerfile` 和 `Dockerfile.debug` 的最终运行阶段均在复制应用和 Python 依赖后执行：

```sh
python scripts/ca_guard.py --apply-system --verify
```

- 解析 `/usr/share/ca-certificates` 中配置启用的证书，按 SHA-256 DER 指纹禁用相关根，而非按文件名或 PEM 注释判断。
- 使用 `update-ca-certificates --fresh` 重建系统 bundle 与 OpenSSL 哈希链接目录，避免只更新单文件。
- 单独验收系统 bundle、系统/默认 capath、默认 cafile 与 certifi；任何已知问题根、缺失/空证书目录、损坏证书都会使构建失败。
- 不添加根证书，不设置 `verify=False`，不使用环境变量掩盖未清理的目录。
- 公共根指纹来源：<https://emudhra.we-have-a-problem.com/roots.json>。测试 fixture 仅含公开证书，没有私钥。

离线测试：

```sh
python -m unittest discover -s tests -p test_ca_guard.py -v
python -m unittest discover -s tests -p test_verify_tls.py -v
```

构建成功还须在实际镜像中验证全部 12 条公开问题链，以及正常 HTTPS 对照；仅将明确的证书信任错误视为拒绝，DNS/超时不能算修复成功。应分别检查默认 `ssl`、只启用 `capath`、显式 certifi 和实际 HTTP 客户端。

该措施覆盖列出的已知根，不承诺阻止业务代码自行指定其他 CA、使用其他内嵌运行时或关闭验证。基础镜像更新后必须再次验收。

已有 systemd/venv 部署并不读取 Dockerfile：若其信任库已经验证通过，不需要因更新 Docker 构建文件而重启。部署时不要覆盖服务器未提交的业务改动或将 `.env`、数据库、虚拟环境、账号数据打包进镜像。

## 来源核对（2026-10-23）

- [CVE 官方记录](https://www.cve.org/CVERecord?id=CVE-2026-111943)对应 Go 标准库 `crypto/x509` 的证书拒绝列表问题，列出 Go 1.26.10 / 1.27.3 修复边界；不能把它直接当作 Python、Node 或整个 CA 事件的统一漏洞编号/版本依据。
- 已对照披露站的 `roots.json`：本项目的 4 个禁用指纹与该列表一致。此为公开披露来源的核对，不替代各平台维护者公告。
- 已对照 PyPI 的 certifi 2026.10.19 发布元数据：`uv.lock` 中 wheel 与 sdist 的 SHA-256 均匹配。版本号检查之外仍保留实际证书指纹检查。

## 在线验收与判定边界

在实际构建出的 Linux 容器中执行（需要能访问公开测试站和正常对照站）：

```sh
docker exec zoaholic python scripts/verify_tls.py
```

脚本先执行离线信任库检查，再执行 5 种客户端模式 × 14 个站点（12 个问题链、2 个正常对照）的检查。只有全部符合预期才退出 0；信任库检查失败、问题链被接受、正常站点失败或结果不确定均不会通过。JSON 的 `ok` 字段与退出码都应检查。

- `urllib` / HTTPX 不跟随重定向；原始站点返回任何 HTTP 状态（含 3xx、4xx、5xx）均意味着 TLS 已被接受，而非证书已被拒绝。
- 超时、DNS 错误、连接失败、过期证书、主机名不匹配不能作为目标根已移除的证据。不能用 `except Exception: 已修复` 判定。
- PEM 主体是 Base64 编码，不能通过搜索可读证书名称验证移除结果；这里解析证书并比较 DER SHA-256 指纹。
- 单纯 `import certifi` 不会让 `urllib` 使用 certifi；脚本的 certifi 模式显式传入其 CA 文件。
- OpenSSL 的 capath 是正常信任库机制，不是后门；只替换 bundle 而不清理 capath 会留下另一条信任路径。
- 根级移除影响该根下的其他合法证书。正常对照站通过不等于所有业务上游兼容，上线前仍应验证实际供应商与代理路径。HTTPX / urllib 可能使用环境代理，原始 socket 模式则直连；代理或防火墙导致失败时须排查，不能跳过验证。

## 仓库与构建上下文卫生

`.gitignore` 保留 `tests/`、`frontend/test/` 的正式测试及公开证书 fixture；临时实验放入明确的 `local/`、`scratch/` 子目录或根目录 `.tmp/`。不要用忽略全部 `test*` 的方式隐藏回归测试。依赖锁文件必须保留。

`.dockerignore` 另外排除工作树、本地工具状态、嵌套依赖/报告及私有插件目录，避免 `COPY . .` 将本地文件放进镜像。忽略规则不能删除已提交的历史内容；可用 `git ls-files -ci --exclude-standard` 检查已跟踪但应忽略的文件。

前端构建仍使用 `node:20-slim`，运行镜像不含该 Node 阶段；后续应单独安排升级到受支持的 LTS 并验证前端构建。Python builder、前端 builder 与 uv 镜像拉取/依赖下载阶段也不受最终运行阶段的 CA 检查保护，不能将运行镜像的验收结论扩展到整个构建链。
