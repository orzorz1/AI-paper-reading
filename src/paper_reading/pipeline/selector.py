"""关键图选择模块。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import OpenAICompatibleClient
from ..models import FigureCandidate, SelectedFigure
from ..utils import truncate_text


class FigureSelectionResponse(BaseModel):
    """选图阶段模型返回结构。"""

    selected_items: list[SelectedFigure] = Field(default_factory=list)


class FigureSelector:
    """调用文本模型，从候选图中选出最关键的 1-3 张图。"""

    def __init__(self, client: OpenAICompatibleClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def select(
        self,
        *,
        title: str,
        abstract: str,
        full_text: str,
        candidates: list[FigureCandidate],
        max_figures: int,
    ) -> list[SelectedFigure]:
        candidate_lines = [
            f"- {item.normalized_id} | 页码={item.page} | caption={item.caption_text}"
            for item in candidates
        ]
        system_prompt = (
            "你是一个帮助普通读者读懂 AI 论文的助手。"
            "你的唯一任务是从候选图里选出最值得读者看的关键图。"
            "必须只返回 JSON，不要解释。"
            "默认优先选 2 张图，只有第三张图明显有助于理解时才选 3 张。"
        )
        user_prompt = (
            f"论文标题：{title}\n\n"
            f"论文摘要：{abstract}\n\n"
            f"论文正文摘要：{truncate_text(full_text, 18000)}\n\n"
            f"候选图列表：\n" + "\n".join(candidate_lines) + "\n\n"
            f"请从候选图中选出最多 {max_figures} 张图。\n"
            "优先选择最能帮助读者理解问题定义、方法主流程和核心结论的图。\n"
            "如果候选里有实验图，只有在性能提升很明显时才选；否则不要为了结果单独多选一张图。\n"
            "禁止输出候选列表里不存在的编号。\n"
            "返回格式：\n"
            "{\n"
            '  "selected_items": [\n'
            '    {"normalized_id": "Fig1", "reason": "一句中文理由", "importance_rank": 1}\n'
            "  ]\n"
            "}"
        )
        response = self._client.complete_json(
            model=self._model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=FigureSelectionResponse,
        )
        return sorted(response.selected_items, key=lambda item: item.importance_rank)
