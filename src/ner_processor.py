"""
NER Processor for BHL Dashboard.
Extracts entities using spaCy, GLiNER2 and other tools
"""

import spacy
from gliner2 import GLiNER2
from collections import Counter
import traceback


LABEL_STANDARDISATION = {
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "DATE": "DATE",
    "SPECIES": "SPECIES",
    "HABITAT": "HABITAT",
    "ORGANISM": "ORGANISM",
    "GENUS": "GENUS",
    "FAMILY": "FAMILY",
    "ORDER": "ORDER",
}

def standardise_label(label):
    """Standardise entity labels for consistency across models."""
    return LABEL_STANDARDISATION.get(label, label)


class NERProcessor:
    """
    Process text using spaCy and GLiNER2.
    Supports three modes: spacy_only, gliner2_only, and hybrid.
    """

    def __init__(self):
        """Initialize both NER models with error handling."""
        print("Loading NER models...")

        # Load spaCy
        self.nlp = None
        try:
            print("  Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            print("  ✓ spaCy model loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading spaCy: {e}")
            print("  spaCy will be disabled for this session.")

        # Load GLiNER2
        self.gliner = None
        try:
            print("  Loading GLiNER2 model...")
            self.gliner = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
            print("  ✓ GLiNER2 model loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading GLiNER2: {e}")
            print("  GLiNER2 will be disabled for this session.")

        # Report status
        if self.nlp is None and self.gliner is None:
            print("\n⚠ WARNING: No models loaded. Processor will not work.")
        elif self.nlp is None:
            print("\n⚠ spaCy disabled. Only GLiNER2 will be used.")
        elif self.gliner is None:
            print("\n⚠ GLiNER2 disabled. Only spaCy will be used.")
        else:
            print("\n✓ Both models loaded successfully.")

    def normalise_text(self, text):
        """Normalise text for consistent processing."""
        if not text or not isinstance(text, str):
            return ""
        
        text = " ".join(text.split())
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")

        return " ".join(text.split())
    
    def process_spacy(self, text):
        """Extract standard entities using spaCy."""
        if self.nlp is None or not text:
            return []

        try:
            doc = self.nlp(text)
            entities = []
            labels = ["PERSON", "GPE", "DATE", "ORG"]

            for ent in doc.ents:
                if ent.label_ in labels:
                    entities.append({
                        "text": ent.text,
                        "label": standardise_label(ent.label_),
                        "source": "spaCy",
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
            return entities
        except Exception as e:
            print(f"  ✗ Error during spaCy processing: {e}")
            return []

    def process_gliner2(self, text):
        """Extract custom entities using GLiNER2 zero-shot."""
        if self.gliner is None or not text:
            return []

        try:
            labels = ["species", "habitat", "location", "date", "person", "organism"]
            predictions = self.gliner.extract_entities(text, labels, threshold=0.3)

            entities = []

            # Handle GLiNER2 output format
            if isinstance(predictions, dict) and "entities" in predictions:
                # Primary format: {"entities": {"species": ["text1", "text2"], ...}}
                entities_dict = predictions["entities"]
                for label, text_list in entities_dict.items():
                    if text_list and isinstance(text_list, list):
                        for entity_text in text_list:
                            if entity_text:
                                start_idx = text.lower().find(entity_text.lower())
                                if start_idx != -1:
                                    end_idx = start_idx + len(entity_text)
                                    entities.append({
                                        "text": entity_text,
                                        "label": standardise_label(label.upper()),
                                        "score": 1.0,
                                        "source": "GLiNER2",
                                        "start": start_idx,
                                        "end": end_idx
                                    })
            elif isinstance(predictions, dict):
                # Fallback: direct label keys
                for label, text_list in predictions.items():
                    if isinstance(text_list, list):
                        for entity_text in text_list:
                            if entity_text:
                                start_idx = text.lower().find(entity_text.lower())
                                if start_idx != -1:
                                    end_idx = start_idx + len(entity_text)
                                    entities.append({
                                        "text": entity_text,
                                        "label": standardise_label(label.upper()),
                                        "score": 1.0,
                                        "source": "GLiNER2",
                                        "start": start_idx,
                                        "end": end_idx
                                    })

            print(f"  GLiNER2 found {len(entities)} entities")
            return entities

        except Exception as e:
            print(f"  ✗ Error during GLiNER2 processing: {e}")
            return []

    def process_text(self, text, method="hybrid"):
        """
        Process text with selected NER method.

        Consistency rules applied:
        1. Text normalisation
        2. Label standardisation
        3. Deduplication

        Args:
            text (str): Input text
            method (str): "spacy_only", "gliner2_only", or "hybrid"

        Returns:
            dict: Combined results
        """
        text = self.normalise_text(text)
        
        if not text or not isinstance(text, str):
            return {
                "entities": [],
                "counts": {},
                "total": 0,
                "spacy_count": 0,
                "gliner_count": 0,
                "error": "Invalid input"
            }

        print("\n" + "=" * 60)
        print(f"Processing Text (Method: {method})")
        print("=" * 60)
        print(f"Text length: {len(text)} characters\n")

        try:
            spacy_entities = []
            gliner_entities = []

            method_lower = method.lower().strip()

            if method_lower == "spacy only" or method_lower == "spacy_only":
                print("Running spaCy only...")
                spacy_entities = self.process_spacy(text)
                print(f"  spaCy found {len(spacy_entities)} entities")
                all_entities = spacy_entities

            elif method_lower == "gliner2 only" or method_lower == "gliner2_only":
                print("Running GLiNER2 only...")
                gliner_entities = self.process_gliner2(text)
                print(f"  GLiNER2 found {len(gliner_entities)} entities")
                all_entities = gliner_entities

            else:  # hybrid (default)
                print("Running spaCy + GLiNER2 hybrid...")
                spacy_entities = self.process_spacy(text)
                print(f"  spaCy found {len(spacy_entities)} entities")
                gliner_entities = self.process_gliner2(text)
                print(f"  GLiNER2 found {len(gliner_entities)} entities")
                all_entities = spacy_entities + gliner_entities

            # Deduplicate
            seen = set()
            unique_entities = []
            for ent in all_entities:
                key = f"{ent['text'].lower()}|{ent['label']}"
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(ent)

            label_counts = Counter([ent["label"] for ent in unique_entities])

            return {
                "entities": unique_entities,
                "counts": dict(label_counts),
                "total": len(unique_entities),
                "spacy_count": len(spacy_entities),
                "gliner_count": len(gliner_entities),
                "method": method
            }
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            print(traceback.format_exc())
            return {
                "entities": [],
                "counts": {},
                "total": 0,
                "spacy_count": 0,
                "gliner_count": 0,
                "error": str(e)
            }


if __name__ == "__main__":
    print("-" * 60)
    print("NER Processor Test")
    print("-" * 60)