# app/__init__.py
from flask import Flask
from flask_cors import CORS
from .core.config import Config
from .api.analyze_routes import bp
from .api.storage_routes import storage_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Client-Id",
        ],
        supports_credentials=False,
        max_age=86400,
    )

    app.register_blueprint(bp, url_prefix="/api")
    app.register_blueprint(storage_bp, url_prefix="/api")

    return app