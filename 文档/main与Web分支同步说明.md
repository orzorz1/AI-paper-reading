# main 与 Web 分支同步说明

本文说明命令行主线（`main`）与带 Web 前端的 `deploy/web-frontend` 分支如何对齐算法改动，以及合并时 `orchestrator.py` 的差异与冲突风险。

> **说明**：下文关于「两分支文件差异」的描述，以文档编写时 `git diff main..deploy/web-frontend -- src/paper_reading/pipeline/orchestrator.py` 的结果为准；若之后任一分支继续修改该文件，请以实际 `git diff` 为准。

## 1. 分支角色

| 分支 | 角色 |
|------|------|
| `main` | 命令行版为主；核心流水线在 `src/paper_reading/pipeline/` 等模块。 |
| `deploy/web-frontend` | 在 `main` 基础上增加 `src/paper_reading/web/`（FastAPI + 静态页），并包含 Web 所需的编排器扩展。 |

Web 端应通过 **`PaperReadingOrchestrator` + `BuildOptions`** 调用与 CLI 相同的 pipeline，避免在 `server.py` 中复制算法逻辑。

## 2. 把 main 上的算法改动同步到 Web 分支

推荐流程：

1. 在 **`main`** 上完成 pipeline / 编排等与「读论文算法」相关的修改。
2. 切换到 **`deploy/web-frontend`**。
3. 执行合并（任选其一）：
   - **常规**：`git merge main`
   - **仅带部分提交**：`git cherry-pick <commit>…`

需要更线性的历史时，可在 Web 分支上 `git rebase main`（冲突需当场解决）。

## 3. Web 如何接入编排器（便于对照）

`src/paper_reading/web/server.py` 中：

- 从 `..pipeline.orchestrator` 导入 `PaperReadingOrchestrator`；
- 构造 `BuildOptions` 后调用 `orchestrator.build(options)`，与 CLI 中 `_build_single` 的路径一致。

因此：**在 `main` 上修改 `pipeline/`、`orchestrator.py`、`models.py` 等，合并进 Web 分支后，Web 会自然使用新行为**（需解决合并冲突时除外）。

### 3.1 Web 与 CLI 的表单行为差异（非两套算法）

在未开启「自定义 API」时，Web 会将 `focus`、`length` 固定为 `method`、`medium`（见 `server.py` 中对应分支）。这是 Web 侧产品策略，与 pipeline 是否共用无关。若希望与 CLI 默认行为完全一致，应改 Web 的表单/默认值逻辑，而不是复制 pipeline。

## 4. `orchestrator.py`：两分支当前差异

`deploy/web-frontend` 相对 `main` 仅多出以下与 Web「自定义 API」相关的改动（其余内容与 `main` 一致）：

1. **import**  
   - Web：`from ..config import AppConfig, OpenAISettings`  
   - `main`：仅 `AppConfig`

2. **类方法**（位于 `__init__` 末尾与 `build` 之间）  
   - Web 独有：`with_openai_settings(self, openai: OpenAISettings) -> "PaperReadingOrchestrator"`  
   - 作用：复用版面解析、PDF 解析等非 LLM 组件，仅替换大模型客户端及相关调用链（供 `server.py` 的 `_orchestrator_for_request` 使用）。

## 5. 合并时最容易冲突的位置

| 区域 | 风险说明 |
|------|----------|
| 文件顶部 `from ..config import …` | `main` 若修改同一行 import，易与 Web 多出的 `OpenAISettings` 在同一 hunk 冲突。 |
| `__init__` 末尾 → `def build` 之间 | Web 在此插入了 `with_openai_settings`；`main` 若在相邻位置增删代码，易冲突。 |
| `build` 及之后 | 当前两分支无差异；日后 `main` 大改 `build` 时，通常仍可与 Web 独占方法自动合并，除非改动紧贴上述插入区边界。 |

## 6. 合并原则（`main` 合入 `deploy/web-frontend` 时）

- **必须保留** Web 分支上的 **`OpenAISettings` 导入** 与 **`with_openai_settings` 整段实现**，否则 Web 自定义 API 路径会失效。
- 将 **`main` 上对 `build`、pipeline 逻辑及其它方法的修改** 合入；若产生冲突，目标结果是：**`main` 的算法与行为更新 + Web 的 `with_openai_settings` + 正确的 import**。

为减少冲突，在 `main` 上优先修改 **`build` 内部或更靠后的私有方法**，尽量避免在「`__init__` 最后一行」与「`def build`」之间插入代码。

## 7. 合并后自检建议

- 在 Web 分支运行测试：`pytest`（或项目约定的测试命令）。
- 本地启动 Web，分别验证：默认 API、自定义 API、上传 PDF 生成结果是否正常。
