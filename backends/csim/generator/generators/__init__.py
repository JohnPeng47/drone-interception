__all__ = [
    "RobustInterceptConfigGenerator",
    "RobustInterceptUniformDistanceConfigGenerator",
]


def __getattr__(name: str):
    if name == "RobustInterceptConfigGenerator":
        from .robust_intercept import RobustInterceptConfigGenerator

        return RobustInterceptConfigGenerator
    if name == "RobustInterceptUniformDistanceConfigGenerator":
        from .robust_intercept_uniform_distance import RobustInterceptUniformDistanceConfigGenerator

        return RobustInterceptUniformDistanceConfigGenerator
    raise AttributeError(name)
