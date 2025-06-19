from .routes import router as main_routes
from .model_api import router as model_routes

all_routes = [main_routes, model_routes]
