"""报告结构规划模块。"""

from __future__ import annotations

from typing import Optional

from ..llm import OpenAICompatibleClient
from ..models import ContentFocus, OutputLength, ReportStructure, WritingStyle
from ..utils import truncate_text


class ReportStructurePlanner:
    """先为文章规划一份更合适的报告结构。"""

    def __init__(
        self,
        client: Optional[OpenAICompatibleClient] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._client = client
        self._model_name = model_name

    def plan(
        self,
        *,
        title: str,
        abstract: str,
        paper_context: str,
        content_focus: ContentFocus,
        output_length: OutputLength,
        writing_style: WritingStyle,
    ) -> ReportStructure:
        if self._client is None or not self._model_name:
            return ReportStructure()

        system_prompt = (
            "你要先为一篇文章规划最终报告的结构。"
            "默认结构来自技术论文，但你不能机械套用。"
            "你要根据文章实际类型，规划一组更贴切的小节结构，"
            "包括非标准技术文章、人文社科文章、综述、理论分析或案例研究。"
            "只能依据给定文章内容判断，不能额外补充文中没有明确说过的结论。"
            "请严格返回 JSON。"
        )
        style_instruction = {
            "professional": "标题和引导语要更书面、稳健、概括性强。",
            "colloquial": "标题和引导语可以更易懂、更接近日常表达，但仍要保持正式文档可读性。",
        }[writing_style]
        user_prompt = (
            f"文章标题：{title}\n\n"
            f"文章摘要：{abstract}\n\n"
            f"内容偏好：{content_focus}\n"
            f"篇幅：{output_length}\n\n"
            f"风格：{writing_style}\n"
            f"{style_instruction}\n\n"
            "默认输出字段和默认结构提示如下：\n"
            "- title_translation\n"
            "- one_sentence_summary\n"
            "- sections:\n"
            "  - motivation -> 默认章节名：研究背景与任务定义\n"
            "  - method_core -> 默认章节名：方法设计与核心机制\n"
            "  - result_summary -> 默认章节名：实验结果与结论\n\n"
            f"文章上下文：{truncate_text(paper_context, 24000)}\n\n"
            "请判断这篇文章是否适合沿用默认结构。"
            "如果不完全适合，请把正文结构改造成更贴合文章内容的 sections。"
            "sections 的数量建议为 2 到 4 个，要求标题自然、书面、概括性强。"
            "key 需要是稳定英文标识，例如 motivation、method_core、result_summary、analysis、findings、discussion、background、case_study。"
            "如果文章不属于典型实验论文，不要硬保留“方法/实验”字样。"
            "例如：人文社科文章可以使用“问题提出与研究背景 / 材料、视角与分析路径 / 主要发现与讨论”；"
            "综述可以使用“研究脉络 / 分类框架 / 挑战与趋势”。\n\n"
            "返回格式：\n"
            "{\n"
            '  "title_translation_label": "title_translation",\n'
            '  "one_sentence_summary_label": "one_sentence_summary",\n'
            '  "sections": [\n'
            '    {"key": "motivation", "title": "研究背景与任务定义", "guidance": "这一节应该写什么"},\n'
            '    {"key": "method_core", "title": "方法设计与核心机制", "guidance": "这一节应该写什么"},\n'
            '    {"key": "result_summary", "title": "实验结果与结论", "guidance": "这一节应该写什么"}\n'
            "  ],\n"
            '  "structure_rationale": "简短说明为什么这样安排结构"\n'
            "}"
        )
        return self._client.complete_json(
            model=self._model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ReportStructure,
        )
