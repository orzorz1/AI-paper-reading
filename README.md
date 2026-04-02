# AI-paper-reading

面向 AI 论文的读图解读工具：提供**命令行**与 **Web 页面**（上传 PDF、下载解读 PDF）。

## 安装

```bash
pip install -e .
```

依赖包含 `doclayout-yolo`、版面识别与 FastAPI 等。若缺少系统库导致 OpenCV 报错（如 `libGL.so.1`），在 Linux 上可安装 `mesa-libGL` 等（见下文「服务器部署要点」）。

## 配置

复制环境变量模板并编辑：

```bash
cp .env.example .env
```

`.env` 中主要变量如下（勿将含密钥的文件提交到仓库）。

```env
# OpenAI 兼容接口
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
TEXT_MODEL=gpt-5.4
VISION_MODEL=gpt-5.4
OPENAI_MAX_RETRIES=5
OPENAI_RETRY_BACKOFF_SECONDS=1.5

# 版面识别模型
LAYOUT_MODEL_PATH=./models/doclayout_yolo_docstructbench_imgsz1024.pt
LAYOUT_DEVICE=cpu
LAYOUT_CONFIDENCE=0.1
LAYOUT_PREDICT_IMGSZ=1024

# 运行时默认配置
OUTPUT_ROOT=./output
DEFAULT_LANG=zh-CN

# Web 版：每次任务输出目录（保留 input.pdf、paper_readable.pdf、artifacts 等）
WEB_OUTPUT_ROOT=./web_output
```

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | 大模型 API Key |
| `OPENAI_BASE_URL` | OpenAI 兼容接口地址 |
| `TEXT_MODEL` / `VISION_MODEL` | 文本 / 多模态模型，用于选图、规划、写作与读图 |
| `OPENAI_MAX_RETRIES` / `OPENAI_RETRY_BACKOFF_SECONDS` | 请求失败时的重试与退避 |
| `LAYOUT_MODEL_PATH` | 版面模型路径（本地 `.pt` 或 Hugging Face 仓库名） |
| `LAYOUT_DEVICE` | 如 `cpu` 或 GPU 编号 `0` |
| `LAYOUT_CONFIDENCE` / `LAYOUT_PREDICT_IMGSZ` | 版面识别阈值与推理尺寸 |
| `OUTPUT_ROOT` | 兼容保留；CLI 默认输出在 PDF 同目录下的时间戳文件夹 |
| `DEFAULT_LANG` | 当前仅支持 `zh-CN` |
| `WEB_OUTPUT_ROOT` | 仅 Web：每次生成结果保存的根目录 |

## 版面识别模型下载

当前使用 **DocLayout-YOLO**。

- 官方项目：<https://github.com/opendatalab/DocLayout-YOLO>
- 推荐权重：<https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench>（如 `doclayout_yolo_docstructbench_imgsz1024.pt`）

推荐目录结构：

```text
AI-paper-reading/
  models/
    doclayout_yolo_docstructbench_imgsz1024.pt
```

`.env` 示例：

```env
LAYOUT_MODEL_PATH=./models/doclayout_yolo_docstructbench_imgsz1024.pt
```

---

## Web 界面

在项目根目录（能读取到 `.env`）执行：

```bash
paper-read-web
```

默认监听 `http://127.0.0.1:8765`。对外访问：

```bash
paper-read-web --host 0.0.0.0 --port 8765
```

### Web 行为摘要

- 默认使用服务端 `.env` 中的大模型与版面配置；可选「使用自定义 API」在浏览器侧填写 Base URL、Key 与模型名（覆盖当次请求的服务端默认）。
- 未开启自定义 API 时，篇幅与解读重心固定为「标准 + 方法学脉络」；开启后可自选。
- 单文件上传上限 **10MB**；同一时刻仅处理 **一个** 生成任务，其他请求会返回「请稍后再试」。
- 每次任务会在 `WEB_OUTPUT_ROOT` 下生成独立子目录，保留 `input.pdf`、`paper_readable.pdf` 及 `artifacts/` 等。

### 域名与反向代理（可选）

用 Nginx 等将 `80`/`443` 反向代理到本机 `127.0.0.1:8765` 即可；HTTPS 建议使用 Let’s Encrypt 或云厂商证书。`paper-read-web` 可改为只监听 `127.0.0.1` 以提高安全性。

---

## 命令行用法

单篇：

```bash
paper-read build /path/to/paper.pdf
```

带偏好：

```bash
paper-read build /path/to/paper.pdf --focus method --length medium
paper-read build /path/to/paper.pdf --focus experiment --length short
```

批量（目录下这一层所有 PDF）：

```bash
paper-read batch /path/to/folder
```

- `--focus`：`method` 或 `experiment`
- `--length`：`short`、`medium`、`long`  
默认：`--focus method`、`--length medium`。

## 输出（CLI）

默认在**输入 PDF 所在目录**下创建带时间戳的文件夹，例如 `paper/test-202604012200/`，内含：

- `paper_readable.md`
- `paper_readable.pdf`
- `artifacts/`（中间产物）

---

## 服务器部署要点

1. **Python**：建议 3.9+，可用 Miniconda 单独建环境后 `pip install -e .`。
2. **Linux 无桌面**：若 `import cv2` 报缺少 `libGL.so.1`，可安装 `mesa-libGL`（发行版包名可能略有差异）。
3. **内存**：版面模型与 PyTorch 占用较高，小内存机器建议加 swap 或限制并发；Web 端已用全局锁避免并行跑多个任务。
4. **常驻进程**：可用 `screen` / `tmux` 或 `systemd` 管理 `paper-read-web`，重启后需重新启动服务。

## 开发与测试

```bash
pip install -e ".[test]"
pytest
```
