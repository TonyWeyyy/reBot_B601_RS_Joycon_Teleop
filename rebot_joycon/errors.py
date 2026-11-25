class RebotError(RuntimeError):
    """项目基础异常。"""


class SafetyError(RebotError):
    """安全校验失败；不能把失败当作空反馈继续运动。"""


class SafeStopRequested(RebotError):
    """用户或安全层请求统一安全回零。"""


class BackendUnavailable(RebotError):
    """指定硬件后端或其依赖不可用。"""
