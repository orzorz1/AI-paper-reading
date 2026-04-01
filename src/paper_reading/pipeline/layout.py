"""版面识别模块。"""

from __future__ import annotations

from pathlib import Path
import re

from PIL import Image

from ..errors import ConfigurationError, DependencyMissingError
from ..models import BoundingBox, LayoutRegion


class LayoutDetector:
    """DocLayout-YOLO 等版面识别模型的统一封装。"""

    def __init__(
        self,
        *,
        model_path: str,
        confidence: float = 0.1,
        device: str = "cpu",
        predict_imgsz: int = 1024,
    ) -> None:
        if not model_path:
            raise ConfigurationError("缺少版面识别模型路径，请设置 LAYOUT_MODEL_PATH 或配置文件 layout.model_path。")
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as exc:
            missing_name = getattr(exc, "name", "")
            if missing_name == "huggingface_hub":
                raise DependencyMissingError(
                    "缺少 huggingface_hub。请安装 `huggingface-hub` 后再运行，"
                    "因为官方 doclayout_yolo 在加载 YOLOv10 时依赖它。"
                ) from exc
            raise DependencyMissingError(
                "缺少 doclayout-yolo。请安装 `doclayout-yolo` 及其依赖后再运行。"
            ) from exc

        self._confidence = confidence
        self._device = device
        self._predict_imgsz = predict_imgsz
        self._model = _load_doclayout_model(YOLOv10, model_path)

    def detect(self, page_image: Path, page_number: int, *, page_width: float, page_height: float) -> list[LayoutRegion]:
        """对单页图片做版面识别。"""
        results = self._model.predict(
            str(page_image),
            imgsz=self._predict_imgsz,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        regions: list[LayoutRegion] = []
        if not results:
            return regions
        with Image.open(page_image) as image:
            image_width, image_height = image.size
        scale_x = page_width / float(image_width or 1)
        scale_y = page_height / float(image_height or 1)
        names = results[0].names
        for box in results[0].boxes:
            raw_label = names.get(int(box.cls[0]), str(int(box.cls[0])))
            label = normalize_layout_label(raw_label)
            if label is None:
                continue
            xyxy = box.xyxy[0].tolist()
            # 模型看到的是渲染后的页面图片，输出坐标单位是像素。
            # 后续匹配和裁图全部基于 PDF 坐标，因此必须先按页面缩放比例映射回 PDF 坐标。
            pdf_bbox = [
                xyxy[0] * scale_x,
                xyxy[1] * scale_y,
                xyxy[2] * scale_x,
                xyxy[3] * scale_y,
            ]
            regions.append(
                LayoutRegion(
                    page=page_number,
                    bbox=BoundingBox.from_sequence(pdf_bbox),
                    label=label,
                    score=float(box.conf[0]),
                )
            )
        return regions


def _load_doclayout_model(yolo_class: type, model_path: str):
    """按官方推荐方式加载 DocLayout-YOLO。

    支持两种输入：
    1. 本地 .pt 权重路径 -> YOLOv10(path)
    2. Hugging Face 仓库名 -> YOLOv10.from_pretrained(repo_id)
    """
    local_path = Path(model_path)
    if local_path.exists():
        return yolo_class(str(local_path))
    if _looks_like_huggingface_repo(model_path):
        return yolo_class.from_pretrained(model_path)
    raise ConfigurationError(
        "LAYOUT_MODEL_PATH 既不是本地模型文件，也不是 Hugging Face 仓库名。"
        "例如：/abs/path/doclayout_yolo_docstructbench_imgsz1024.pt "
        "或 juliozhao/DocLayout-YOLO-DocStructBench"
    )


def _looks_like_huggingface_repo(model_path: str) -> bool:
    """判断输入是否像 Hugging Face repo_id。"""
    if Path(model_path).suffix.lower() in {".pt", ".onnx", ".pth"}:
        return False
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", model_path.strip()))


def normalize_layout_label(raw_label: str) -> str | None:
    """把模型输出的类别名归一到工具内部使用的四种标签。"""
    normalized = raw_label.strip().lower().replace("_", " ").replace("-", " ")
    if "caption" in normalized:
        return "caption"
    if "table" in normalized:
        return "table"
    if "figure" in normalized or normalized == "fig":
        return "figure"
    if "text" in normalized or "paragraph" in normalized or "plain" in normalized:
        return "text"
    return None
