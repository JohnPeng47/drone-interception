__all__ = ["RobustInterceptConfigGenerator"]


def __getattr__(name: str):
    if name == "RobustInterceptConfigGenerator":
        from .robust_intercept import RobustInterceptConfigGenerator

        return RobustInterceptConfigGenerator
    raise AttributeError(name)
