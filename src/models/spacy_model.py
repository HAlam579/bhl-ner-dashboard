"""
EXAMPLE: How to wrap a model for the plugin system.
"""

from .base import BaseNERModel
import spacy


class SpacyModel(BaseNERModel):
    
    @property
    def name(self):
        return "spaCy"
    
    @property
    def key(self):
        return "spacy"
    
    def load(self):
        self.model = spacy.load("en_core_web_sm")
        self.loaded = True
    
    def extract(self, text):
        if not self.loaded:
            return []
        
        doc = self.model(text)
        entities = []
        labels = ["PERSON", "GPE", "DATE", "ORG"]
        
        for ent in doc.ents:
            if ent.label_ in labels:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "source": self.name,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "score": None
                })
        return entities