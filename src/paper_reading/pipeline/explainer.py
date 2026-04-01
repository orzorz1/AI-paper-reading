"""逐图解释模块。"""

from __future__ import annotations

from pathlib import Path

from ..llm import OpenAICompatibleClient
from ..models import FigureCandidate, FigureExplanation
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
    ) -> FigureExplanation:
        system_prompt = (
            "你要把论文里的单张图解释给非专业读者。"
            "输出必须是简体中文 JSON。"
            "不要复述太多原始 caption，要强调这张图应该怎么看。"
            "整体必须简短，适合融入 1 到 2 分钟的 Markdown 阅读短文。"
        )
        user_prompt = (
            f"论文标题：{paper_title}\n\n"
            f"论文摘要：{abstract}\n\n"
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
            "- 语言风格尽量正式，不要太口语化\n\n"
            "- what_it_shows、how_to_read、why_it_matters 这三段会被直接拼进 Markdown，尽量写成自然短句\n"
            "- 不要依赖固定模板句，不要总是用“这张图展示了”“如下图所示”这类开头\n\n"
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
