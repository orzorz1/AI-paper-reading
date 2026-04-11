"""整体阅读结构生成模块。"""

from __future__ import annotations

from ..llm import OpenAICompatibleClient
from ..models import ContentFocus, FigureCandidate, OutputLength, ReportStructure, SelectedFigure, StoryOutline, WritingStyle
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
        report_structure: ReportStructure,
        selected_figures: list[SelectedFigure],
        candidates_by_id: dict[str, FigureCandidate],
        content_focus: ContentFocus,
        output_length: OutputLength,
        writing_style: WritingStyle,
    ) -> StoryOutline:
        figure_context = []
        for item in selected_figures:
            candidate = candidates_by_id[item.normalized_id]
            figure_context.append(
                f"- {item.normalized_id} | 角色提示={item.reason} | 页码={candidate.page} | caption={candidate.caption_text}"
            )
        focus_instruction = {
            "method": "写作重心偏方法，但仍要让读者知道任务是什么。",
            "experiment": "写作重心偏实验，但仍要让读者知道方法核心是什么。",
        }[content_focus]
        length_instruction = {
            "short": "短篇输出，整体更凝练，信息只保留主线。",
            "medium": "中篇输出，保持当前默认信息密度。",
            "long": "长篇输出，内容可以更充实。无论偏向方法还是偏向实验，都要同时覆盖方法和实验。",
        }[output_length]
        formula_instruction = {
            "short": "除非特别必要，否则不要主动引入公式。",
            "medium": "如果论文里有对理解方法、结论或核心论证确实重要的公式，可以保留关键公式，并配一句简短解释；不要写成公式堆砌。",
            "long": "如果论文里存在对理解主线重要的公式，可以保留关键公式，并在正文里配一句通俗解释。",
        }[output_length]
        structure_instruction = {
            ("method", "short"): "最终成文应明显偏向方法，只需保留任务背景和方法主线，结果只在特别关键时一笔带过。",
            ("method", "medium"): "最终成文应以方法为主，背景简洁，方法最完整，结果可选且简短。",
            ("method", "long"): "最终成文必须同时覆盖背景、方法、实验，其中方法部分要明显比实验更详细。",
            ("experiment", "short"): "最终成文应偏向实验，但仍要先交代动机和方法，再进入实验结果。",
            ("experiment", "medium"): "最终成文应以实验为主，但结构上仍需先讲研究背景与方法主线，再讲实验。",
            ("experiment", "long"): "最终成文必须同时覆盖方法和实验，结构上先讲动机和方法，再系统讲实验，其中实验部分要明显比方法更详细。",
        }[(content_focus, output_length)]
        style_instruction = {
            "professional": "标题和正文都要偏书面、凝练、专业，避免太口语化。",
            "colloquial": "标题和正文都要更平实、更好懂，可以更接近日常表达，但不要随意或松散。",
        }[writing_style]

        system_prompt = (
            "你要为一篇 AI 论文生成中文阅读提纲。"
            "这个提纲不是讲稿提纲，而是供最终 Markdown 直接展开成阅读材料的骨架。"
            "请用简体中文、面向非领域读者、严格返回 JSON。"
            "你必须服从给定的内容偏好和篇幅要求。"
            "同时要服从给定的写作风格要求。"
            "不能编造论文里没有明确提到的事实、贡献或结论。"
        )
        section_lines = []
        for index, section in enumerate(report_structure.sections, start=1):
            section_lines.append(
                f"{index}. key={section.key} | 标题={section.title} | 写作引导={section.guidance}"
            )
        user_prompt = (
            f"论文标题：{title}\n\n"
            f"论文摘要：{abstract}\n\n"
            f"内容偏好：{content_focus}\n"
            f"篇幅：{output_length}\n"
            f"风格：{writing_style}\n"
            f"{focus_instruction}\n"
            f"{length_instruction}\n"
            f"{structure_instruction}\n\n"
            f"{formula_instruction}\n\n"
            f"{style_instruction}\n\n"
            f"报告结构建议：\n"
            f"- {report_structure.title_translation_label}\n"
            f"- {report_structure.one_sentence_summary_label}\n"
            f"- sections:\n{chr(10).join(section_lines)}\n"
            f"- 结构说明：{report_structure.structure_rationale}\n\n"
            f"论文上下文：{truncate_text(paper_context, 32000)}\n\n"
            f"正文补充摘录：{truncate_text(full_text, 22000)}\n\n"
            f"关键图：\n{chr(10).join(figure_context)}\n\n"
            "请输出：\n"
            f"1. 中文题目翻译 {report_structure.title_translation_label}，要求自然、准确，不要直译得生硬\n"
            f"2. 导读摘要 {report_structure.one_sentence_summary_label}：短篇控制在 1 句，中篇控制在 2 句内，长篇控制在 2 到 3 句\n"
            "3. sections：请严格按照给定报告结构建议输出同数量的小节。每个小节都要返回 key、title、content。\n"
            "   - title 必须与结构建议中的标题一致\n"
            "   - content 要写成可直接进入 Markdown 的自然短段落，不要写成提纲句\n"
            "   - 短篇：每节 1 到 2 句；中篇：每节 2 到 3 句；长篇：每节 3 到 5 句\n"
            "   - 如果某一节在这篇文章里确实不重要，可以保留但写得简短；不要随意删除 sections\n"
            "4. figure_roles：这是内部字段，用来说明每张图在最终文档里的角色，只能使用 problem、method、result、support 这 4 个标签\n\n"
            "额外要求：\n"
            "- 最终文档不要单独列出“读者应记住什么”\n"
            "- 除非是长篇，否则如果实验提升不大，不要硬写结果部分\n"
            "- 不要把内容写成提纲口吻，要能直接展开成短文\n"
            "- 不要单独输出术语表，术语解释直接写进正文\n"
            "- 对不太常见、但理解论文必须知道的缩写，例如 RES、MRES，只有第一次出现时写成“RES（中文）”这种形式，后面直接写缩写\n"
            "- 对常见缩写，例如 CNN、LLM、SOTA，不要专门解释，直接使用英文缩写\n"
            "- 不要写英文全称，正文里只保留必要的中文括注\n\n"
            "- sections 里的每个 content 都会被直接放进 Markdown，尽量写成自然、完整的短段落\n"
            "- 不要依赖固定模板句，不要总是用“这张图展示了”“如下图所示”这类开头\n\n"
            "- long 模式下，如果关键公式对理解方法或结论不可替代，可以保留 1 到 2 个最重要的公式，并配简短解释；如果公式不关键，就不要硬加\n\n"
            "- 如果需要输出公式，不要使用 ```math 或任何 fenced code block 包裹公式\n"
            "- 行内公式统一写成 $...$，独立公式统一写成 $$...$$\n"
            "- 不要使用 \\(...\\) 或 \\[...\\] 这种写法\n"
            "- 不要把公式写进 Markdown 代码块，也不要输出三引号包裹的公式块\n\n"
            "- 不要擅自写“首次提出”“首次建立”“开创性”“首次实现”等强结论，除非文章上下文里明确写了这类表述\n"
            "- 如果文中只是展示结果或提出方法，就按文中证据如实概括，不要上升成更强的历史性判断\n\n"
            "返回格式：\n"
            "{\n"
            '  "title_translation": "...",\n'
            '  "one_sentence_summary": "...",\n'
            '  "sections": [\n'
            '    {"key": "motivation", "title": "研究背景与任务定义", "content": "..."},\n'
            '    {"key": "method_core", "title": "方法设计与核心机制", "content": "..."}\n'
            "  ],\n"
            '  "figure_roles": {"Fig1": "problem", "Fig4": "method"}\n'
            "}"
        )
        return self._client.complete_json(
            model=self._model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=StoryOutline,
        )
