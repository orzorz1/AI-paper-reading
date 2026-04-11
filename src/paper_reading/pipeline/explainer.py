"""逐图解释模块。"""

from __future__ import annotations

from pathlib import Path

from ..llm import OpenAICompatibleClient
from ..models import ContentFocus, FigureCandidate, FigureExplanation, OutputLength, WritingStyle
from ..utils import truncate_text


class FigureExplainer:
    """用多模态模型解释单张关键图。"""

    def __init__(self, client: OpenAICompatibleClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def explain(
        self,
        *,
        candidate: FigureCandidate,
        figure_role: str,
        paper_title: str,
        abstract: str,
        paper_context: str,
        surrounding_text: str,
        content_focus: ContentFocus,
        output_length: OutputLength,
        writing_style: WritingStyle,
    ) -> FigureExplanation:
        focus_instruction = {
            "method": "解释时更偏向帮助读者理解方法设计和模块作用。",
            "experiment": "解释时更偏向帮助读者理解实验设置、结果含义和对比关系。",
        }[content_focus]
        length_instruction = {
            "short": "整体更凝练，每段尽量压到 1 句。",
            "medium": "保持当前默认长度。",
            "long": "允许稍微展开，但仍然要短，避免流水账。",
        }[output_length]
        formula_instruction = {
            "short": "除非特别必要，否则不要主动引入公式。",
            "medium": "如果这张图和某个关键公式强相关，可以点出公式的作用，不必展开推导，但重点仍然是把图讲明白。",
            "long": "如果这张图和某个关键公式强相关，可以顺带点出该公式的作用，但不要堆公式。",
        }[output_length]
        style_instruction = {
            "professional": "标题和正文都要更书面、专业、稳健。",
            "colloquial": "标题和正文可以更平实、更像直接跟读者解释，但不要过于随意。",
        }[writing_style]
        system_prompt = (
            "你要把论文里的单张图解释给非专业读者。"
            "输出必须是简体中文 JSON。"
            "不要复述太多原始 caption，要强调这张图应该怎么看。"
            "整体必须简短，适合融入给定篇幅的 Markdown 阅读短文。"
            "只能根据给定图片和文章上下文解释，不能编造文中没有明确说过的结论。"
        )
        user_prompt = (
            f"论文标题：{paper_title}\n\n"
            f"论文摘要：{abstract}\n\n"
            f"内容偏好：{content_focus}\n"
            f"篇幅：{output_length}\n"
            f"风格：{writing_style}\n"
            f"{focus_instruction}\n"
            f"{length_instruction}\n\n"
            f"{formula_instruction}\n\n"
            f"{style_instruction}\n\n"
            f"论文上下文：{truncate_text(paper_context, 18000)}\n\n"
            f"图编号：{candidate.normalized_id}\n"
            f"图角色：{figure_role}\n"
            f"图 caption：{candidate.caption_text}\n\n"
            f"图附近正文：{truncate_text(surrounding_text, 12000)}\n\n"
            "请输出：\n"
            "1. 一个适合放在文档里的简短小标题\n"
            "2. 这张图在说什么，控制在 2 句以内\n"
            "3. 读者应该怎么看这张图，控制在 2 句以内\n"
            "4. 这张图为什么重要，控制在 2 句以内\n\n"
            "额外要求：\n"
            "- 不要输出“读者应记住什么”\n"
            "- 如果这是结果图，只简单说明任务和是否有明显提升\n"
            "- 不要逐格念表格，不要长篇复述实验数字\n"
            "- 对不太常见、但理解论文必须知道的缩写，例如 RES、MRES，只有第一次出现时写成“缩写（中文）”，后面直接写缩写\n"
            "- 对常见缩写，例如 CNN、LLM、SOTA，不要专门解释，直接使用英文缩写\n"
            "- 不要写英文全称，正文里只保留必要的中文括注\n"
            "- 必须服从给定风格要求\n\n"
            "- 不要擅自写“首次提出”“首次验证”“首次实现”等表述，除非给定上下文中明确出现\n\n"
            "- what_it_shows、how_to_read、why_it_matters 这三段会被直接拼进 Markdown，尽量写成自然短句\n"
            "- 不要依赖固定模板句，不要总是用“这张图展示了”“如下图所示”这类开头\n\n"
            "- 如果需要提到公式，不要使用 ```math 或任何 fenced code block 包裹公式\n"
            "- 行内公式统一写成 $...$，独立公式统一写成 $$...$$\n"
            "- 不要使用 \\(...\\) 或 \\[...\\] 这种写法\n\n"
            "返回格式：\n"
            "{\n"
            f'  "normalized_id": "{candidate.normalized_id}",\n'
            '  "title": "...",\n'
            '  "what_it_shows": "...",\n'
            '  "how_to_read": "...",\n'
            '  "why_it_matters": "..."\n'
            "}"
        )
        return self._client.complete_vision_json(
            model=self._model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=[Path(candidate.image_path)],
            response_model=FigureExplanation,
        )
