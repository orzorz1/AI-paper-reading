"""项目内统一异常定义。"""


class PaperReadingError(Exception):
    """所有可预期业务异常的基类。"""


class ConfigurationError(PaperReadingError):
    """配置缺失或配置格式错误。"""


class DependencyMissingError(PaperReadingError):
    """运行时依赖不存在。"""


class LLMResponseError(PaperReadingError):
    """模型返回内容不符合预期。"""


class LLMServiceError(PaperReadingError):
    """模型服务请求失败，例如网关错误、超时或限流。"""


class SelectionValidationError(PaperReadingError):
    """关键图选择结果与候选集合不一致。"""


class PipelineExecutionError(PaperReadingError):
    """流水线执行失败。"""
