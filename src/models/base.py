"""
Base class for all NER models.
To add a new model, inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod


class BaseNERModel(ABC):
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def extract(self, text):
        """
        Extract entities from text.
        
        Returns: List of dicts with keys:
            - text: str
            - label: str
            - source: str
            - start: int
            - end: int
            - score: float (optional)
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    @abstractmethod
    def key(self):
        """Unique key for this model (e.g., 'spacy', 'gliner')."""
        pass
    
    @property
    def loaded(self):
        """Check if model is loaded."""
        return getattr(self, '_loaded', False)
    
    @loaded.setter
    def loaded(self, value):
        self._loaded = value