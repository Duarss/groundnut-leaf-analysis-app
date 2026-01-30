# app/__init__.py
from flask import Flask
from flask_cors import CORS
from .core.config import Config
from .api.analyze_routes import bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Izinkan akses dari frontend (Vite dev server)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprint untuk klasifikasi
    app.register_blueprint(bp, url_prefix="/api")

    return app