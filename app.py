import os
from flask import Flask

from api.analyzer_routes import analyzer_bp
from api.feedback_routes import feedback_bp
from api.report_routes import report_bp
from api.history_routes import history_bp


def create_app():
    app = Flask(__name__, static_folder=".", template_folder=".")
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB uploads

    # Register feature blueprints
    app.register_blueprint(analyzer_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(history_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
