"""LSED Models Module."""

# Lazy imports to avoid loading torch when not needed
def __getattr__(name):
    if name == "LSED":
        from .lsed import LSED
        return LSED
    elif name == "IncrementalLSED":
        from .lsed import IncrementalLSED
        return IncrementalLSED
    elif name == "LLMSummarizer":
        from .llm_summarizer import LLMSummarizer
        return LLMSummarizer
    elif name == "PromptTemplates":
        from .llm_summarizer import PromptTemplates
        return PromptTemplates
    elif name == "Vectorizer":
        from .vectorizer import Vectorizer
        return Vectorizer
    elif name == "SBERTVectorizer":
        from .vectorizer import SBERTVectorizer
        return SBERTVectorizer
    elif name == "Word2VecVectorizer":
        from .vectorizer import Word2VecVectorizer
        return Word2VecVectorizer
    elif name == "TimeEncoder":
        from .vectorizer import TimeEncoder
        return TimeEncoder
    elif name == "HyperbolicEncoder":
        from .hyperbolic_encoder import HyperbolicEncoder
        return HyperbolicEncoder
    elif name == "PoincareBall":
        from .hyperbolic_encoder import PoincareBall
        return PoincareBall
    elif name == "Hyperboloid":
        from .hyperbolic_encoder import Hyperboloid
        return Hyperboloid
    elif name == "EventClustering":
        from .clustering import EventClustering
        return EventClustering
    elif name == "HyperbolicKMeans":
        from .clustering import HyperbolicKMeans
        return HyperbolicKMeans
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LSED",
    "IncrementalLSED",
    "LLMSummarizer",
    "PromptTemplates",
    "Vectorizer",
    "SBERTVectorizer",
    "Word2VecVectorizer",
    "TimeEncoder",
    "HyperbolicEncoder",
    "PoincareBall",
    "Hyperboloid",
    "EventClustering",
    "HyperbolicKMeans",
]
