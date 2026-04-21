# AI-paper-reading

网页版：paper.emio.cn

一个面向 AI 论文的小工具。

当前是命令行工具，没有页面。

## 安装

```bash
pip install -e .
```

## 配置

先复制模板文件：

```bash
cp .env.example .env
```

然后按需修改 `.env`。模板文件是：

- `.env.example`

下面是当前项目里用到的完整变量：

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
```

变量说明：

- `OPENAI_API_KEY`：大模型 API key。
- `OPENAI_BASE_URL`：OpenAI 兼容接口地址。
- `TEXT_MODEL`：文本模型，当前用于选图、整体规划、Markdown 写作。
- `VISION_MODEL`：多模态模型，当前用于逐图解释。
- `OPENAI_MAX_RETRIES`：模型请求失败后的最大重试次数。
- `OPENAI_RETRY_BACKOFF_SECONDS`：重试退避时间基数，实际等待时间会按重试轮次递增。
- `LAYOUT_MODEL_PATH`：版面识别模型路径。支持本地 `.pt` 权重，也支持 Hugging Face 仓库名。
- `LAYOUT_DEVICE`：版面识别运行设备，常见值有 `cpu`、`0`。
- `LAYOUT_CONFIDENCE`：版面识别置信度阈值。
- `LAYOUT_PREDICT_IMGSZ`：版面识别推理分辨率。
- `OUTPUT_ROOT`：兼容保留配置。当前默认输出目录不再用它，而是直接输出到输入 PDF 所在目录；如果后面恢复全局输出根目录策略，这个字段可以继续复用。
- `DEFAULT_LANG`：默认输出语言。当前只支持 `zh-CN`。

不要把真实 `.env` 提交到仓库里，尤其是 `OPENAI_API_KEY`。

## 版面识别模型下载

当前版面识别使用 `DocLayout-YOLO`。

推荐下载：

- 官方项目：`https://github.com/opendatalab/DocLayout-YOLO`
- 推荐权重仓库：`https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench`
- 推荐文件名：`doclayout_yolo_docstructbench_imgsz1024.pt`
- 文件页：`https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/blob/main/doclayout_yolo_docstructbench_imgsz1024.pt`

最简单的放置方式：

```text
AI-paper-reading/
  models/
    doclayout_yolo_docstructbench_imgsz1024.pt
```

然后在 `.env` 里这样写：

```env
LAYOUT_MODEL_PATH=./models/doclayout_yolo_docstructbench_imgsz1024.pt
```

## 用法

单篇处理：

```bash
paper-read build /path/to/paper.pdf
```

带生成偏好的示例：

```bash
paper-read build /path/to/paper.pdf --focus method --length medium
paper-read build /path/to/paper.pdf --focus experiment --length short
paper-read build /path/to/paper.pdf --focus experiment --length long
```

批量处理某个目录下这一层的所有 PDF：

```bash
paper-read batch /path/to/folder
```

其中：

- `--focus`：`method` 或 `experiment`
- `--length`：`short`、`medium`、`long`

当前默认是：

- `--focus method`
- `--length medium`

## 输出

默认输出到输入 PDF 所在目录，目录名格式：

```text
原pdf名-时间
```

例如：

```text
paper/test-202604012200/
```

每篇会生成：

- `paper_readable.md`
- `paper_readable.pdf`
- `artifacts/` 中间产物目录
