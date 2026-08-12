try:
    from gloss.gloss import GLOSS
    __all__ = ["GLOSS"]
except ImportError:
    __all__ = []
