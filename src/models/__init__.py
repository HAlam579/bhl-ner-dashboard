import os
import importlib
import inspect
from .base import BaseNERModel


def discover_models():
    """
    Auto-discover all model classes in the models folder.
    
    Returns:
        dict: {model_key: model_class}
    """
    models = {}
    current_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
            module_name = filename[:-3]
            
            try:
                module = importlib.import_module(f'.{module_name}', package=__package__)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseNERModel) and 
                        obj is not BaseNERModel and 
                        hasattr(obj, 'key')):
                        try:
                            temp = obj()
                            key = temp.key
                            models[key] = obj
                        except Exception:
                            pass
            except Exception:
                continue
    
    return models


# Pre-discover models
AVAILABLE_MODELS = discover_models()