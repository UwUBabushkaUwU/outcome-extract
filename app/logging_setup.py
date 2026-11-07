import logging
import os


def setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(processName)s %(name)s: %(message)s",
    )
    # The OpenAI client logs every request at INFO; our own retry logs are enough.
    logging.getLogger("httpx").setLevel(logging.WARNING)
