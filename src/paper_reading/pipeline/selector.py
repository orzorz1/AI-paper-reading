"""关键图选择模块。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..llm import OpenAICompatibleClient
from ..models import ContentFocus, FigureCandidate, OutputLength, SelectedFigure
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
        content_focus: ContentFocus,
        output_length: OutputLength,
        candidate_kind: Literal["figure", "table"] = "figure",
    ) -> list[SelectedFigure]:
        candidate_lines = [
            f"- {item.normalized_id} | 页码={item.page} | caption={item.caption_text}"
            for item in candidates
        ]
        kind_instruction = {
            "figure": "当前阶段只筛选图片（Fig*）。",
            "table": "当前阶段只筛选表格（Table*）。表格不受图片数量上限限制，只按是否真的有助于理解实验来决定保留。",
        }[candidate_kind]
        focus_instruction = {
            "method": "当前内容偏向方法。优先选择能讲清问题设定、方法主流程、关键模块设计的图。",
            "experiment": "当前内容偏向实验。优先选择能讲清评测任务、对比结果、数据或实验设置的图。",
        }[content_focus]
        length_instruction = {
            "short": "这是短篇输出，只选最关键的 1 张图。",
            "medium": "这是中篇输出，优先选 2 张图；只有第三张图明显提升理解时才选 3 张。",
            "long": f"这是长篇输出，可以选到 {max_figures} 张图。无论偏向方法还是偏向实验，都要兼顾方法图和实验图。",
        }[output_length]
        system_prompt = (
            "你是一个帮助普通读者读懂 AI 论文的助手。"
            "你的唯一任务是从候选图里选出最值得读者看的关键图。"
            "必须只返回 JSON，不要解释。"
            "选图时要服从给定的内容偏好和篇幅要求。"
        )
        user_prompt = (
            f"论文标题：{title}\n\n"
            f"论文摘要：{abstract}\n\n"
            f"论文正文摘要：{truncate_text(full_text, 18000)}\n\n"
            f"候选图列表：\n" + "\n".join(candidate_lines) + "\n\n"
            f"内容偏好：{content_focus}\n"
            f"篇幅：{output_length}\n"
            f"筛选阶段：{candidate_kind}\n"
            f"{kind_instruction}\n"
            f"{focus_instruction}\n"
            f"{length_instruction}\n"
            "请从候选里选出真正值得保留的项。\n"
            f"- 本阶段最多返回 {max_figures} 个候选\n"
            "优先选择最能帮助读者理解问题定义、方法主流程和核心结论的图。\n"
            "如果是短篇，尽量只留一张最关键的图。\n"
            "如果是中篇且偏向方法，实验图只有在性能提升很明显时才选；否则不要为了结果单独多选一张图。\n"
            "如果是中篇且偏向实验，可以优先保留一张结果图，但仍要保证读者能看懂论文在做什么。\n"
            "如果是长篇，方法图和实验图都应至少各有一张，只要候选里确实存在。\n"
            "如果当前阶段是表格筛选，只有那些真正有助于理解实验设定、主结果或关键消融的表格才应该保留。\n"
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
