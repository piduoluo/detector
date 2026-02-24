from pathlib import Path

from detector.config import load_config
from detector.webapp import create_app

BASE_DIR = Path(__file__).resolve().parent
CONFIG = load_config(BASE_DIR)

# Export `app` so WSGI servers can use `app:app`.
app = create_app(CONFIG)


if __name__ == "__main__":
    app.run(host=CONFIG.host, port=CONFIG.port, debug=CONFIG.debug)
