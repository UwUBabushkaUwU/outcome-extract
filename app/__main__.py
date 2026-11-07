"""`python -m app`: the API on PORT (8080) plus WORKERS (4) worker processes."""

import logging

import uvicorn

from app import config, db
from app.logging_setup import setup_logging
from app.worker import Supervisor

log = logging.getLogger("app")


def main() -> None:
    setup_logging()
    for d in (config.DATA_DIR, config.FILES_DIR, config.TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    db.init_db()

    supervisor = Supervisor(config.WORKERS)
    supervisor.start()
    log.info("started %d workers; API on %s:%d", config.WORKERS, config.HOST, config.PORT)
    server = uvicorn.Server(uvicorn.Config(
        "app.main:app", host=config.HOST, port=config.PORT, log_config=None,
        timeout_graceful_shutdown=5,
    ))
    try:
        server.run()  # returns on SIGINT / SIGTERM
    finally:
        log.info("shutting down workers")
        supervisor.shutdown()


if __name__ == "__main__":
    main()
