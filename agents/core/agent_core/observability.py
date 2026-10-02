import logging
import os


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        force=True,
    )
    # httpx logs every request at INFO, which drowns out the agent's own logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def setup_tracing(project: str) -> bool:
    """Enable LangSmith tracing when an API key is present.

    Accepts both the current LANGSMITH_* variables and the older LANGCHAIN_*
    names. Without a key tracing is switched off explicitly, so the SDK does
    not try to upload runs and log warnings on every request.
    """
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return False
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT") or project)
    return os.environ["LANGSMITH_TRACING"].lower() == "true"
