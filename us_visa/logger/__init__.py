import logging
import os
import sys
from datetime import datetime

LOG_FORMAT = "[ %(asctime)s ] %(name)s -%(levelname)s - %(message)s"


def _setup_logging() -> None:
    # Vercel's filesystem is read-only (except /tmp), so log to the console there.
    if not os.getenv("VERCEL"):
        try:
            from from_root import from_root

            log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
            log_dir = "logs"
            logs_path = os.path.join(from_root(), log_dir, log_file)
            os.makedirs(os.path.dirname(logs_path), exist_ok=True)

            logging.basicConfig(
                filename=logs_path,
                format=LOG_FORMAT,
                level=logging.DEBUG,
            )
            return
        except Exception:
            pass  # read-only filesystem or root not found -> fall back to console

    logging.basicConfig(
        stream=sys.stdout,
        format=LOG_FORMAT,
        level=logging.INFO,
    )


_setup_logging()