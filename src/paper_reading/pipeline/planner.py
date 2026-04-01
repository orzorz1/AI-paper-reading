"""整体阅读结构生成模块。"""

from __future__ import annotations

from ..llm import OpenAICompatibleClient
from ..models import FigureCandidate, SelectedFigure, StoryOutline
from ..utils import truncate_text


class StoryPlanner:
    """生成最终文档的整体阅读骨架。"""

    def __init__(self, client: OpenAICompatibleClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def plan(
        self,
        *,
        title: str,
        abstract: str,
        full_text: str,
        paper_context: str,
        selected_figures: list[SelectedFigure],
        candidates_by_id: dict[str, FigureCandidate],
    ) -> StoryOutline:
        figure_context = []
        for item in selected_figures:
            candidate = candidates_by_id[item.normalized_id]
            figure_context.append(
                f"- {item.normalized_id} | 角色提示={item.reason} | 页码={candidate.page} | caption={candidate.caption_text}"
            )

        system_prompt = (
            "你要为一篇 AI 论文生成中文阅读提纲。"
            "这个提纲不是讲稿提纲，而是供最终 Markdown 直接展开成阅读材料的骨架。"
            "请用简体中文、面向非领域读者、严格返回 JSON。"
            "整体阅读时间控制在 1 到 2 分钟，内容要比摘要更充实，但不要拖长。"
        )
        user_prompt = (
            f"论文标题：{title}\n\n"
            f"论文摘要：{abstract}\n\n"
            f"论文上下文：{truncate_text(paper_context, 32000)}\n\n"
            f"正文补充摘录：{truncate_text(full_text, 22000)}\n\n"
            f"关键图：\n{chr(10).join(figure_context)}\n\n"
            "请输出：\n"
            "1. 中文题目翻译 title_translation，要求自然、准确，不要直译得生硬\n"
            "2. 导读摘要 one_sentence_summary，控制在 2 句以内，80 到 120 字\n"
            "3. 研究背景与任务定义 motivation，控制在 2 到 3 句\n"
            "4. 方法核心 method_core，控制在 3 到 4 句\n"
            "5. 实验结果 result_summary：只在有明显性能提升或明显新结果时写 1 到 2 句，否则返回空字符串\n"
            "6. figure_roles：这是内部字段，用来说明每张图在最终文档里的角色，只能使用 problem、method、result、support 这 4 个标签\n\n"
            "额外要求：\n"
            "- 最终文档不要单独列出“读者应记住什么”\n"
            "- 如果实验提升不大，不要硬写结果部分\n"
            "- 不要把内容写成提纲口吻，要能直接展开成短文\n"
            "- 不要单独输出术语表，术语解释直接写进正文\n"
            "- 对不太常见、但理解论文必须知道的缩写，例如 RES、MRES，只有第一次出现时写成“RES（中文）”这种形式，后面直接写缩写\n"
            "- 对常见缩写，例如 CNN、LLM、SOTA，不要专门解释，直接使用英文缩写\n"
            "- 不要写英文全称，正文里只保留必要的中文括注\n\n"
            "- motivation、method_core、result_summary 这几段会被直接放进 Markdown，尽量写成自然、完整的短段落\n"
            "- 不要依赖固定模板句，不要总是用“这张图展示了”“如下图所示”这类开头\n\n"
            "返回格式：\n"
            "{\n"
            '  "title_translation": "...",\n'
            '  "one_sentence_summary": "...",\n'
            '  "motivation": "...",\n'
            '  "method_core": "...",\n'
            '  "result_summary": "",\n'
            '  "figure_roles": {"Fig1": "problem", "Fig4": "method"}\n'
            "}"
        )
        return self._client.complete_json(
            model=self._model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=StoryOutline,
        )
