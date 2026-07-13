"""
NER Processor for BHL Dashboard.
Extracts entities using spaCy, GLiNER2 and other tools
"""

import spacy
from gliner2 import GLiNER2
from gliner import GLiNER
from collections import Counter
import traceback
import subprocess
import json
import tempfile
import os


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gliner")


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
        """Initialise the processor. Models are not loaded yet."""
        self.nlp = None
        self.gliner2 = None
        self.gliner = None
        
        self.gnfinder_available = False
        self.gnfinder_path = None
        
        self.models_loaded = {
            "spacy": False,
            "gliner2": False,
            "gliner": False,
            "gnfinder": False

        }

        self._check_gnfinder()

    def _check_gnfinder(self):
        """Check if GNfinder binary is available on the system."""
        # Common locations to check for gnfinder
        gnfinder_paths = [
            r"C:\bin\gnfinder.exe",
            r"C:\bin\gnfinder",
            os.path.join(os.path.dirname(__file__), "gnfinder.exe"),
            os.path.join(os.path.dirname(__file__), "..", "gnfinder.exe"),
            os.path.join(os.path.dirname(__file__), "..", "bin", "gnfinder.exe"),
            os.path.join(os.path.dirname(__file__), "tools", "gnfinder.exe"),
            os.path.join(os.environ.get("USERPROFILE", ""), "bin", "gnfinder.exe"),
            "gnfinder.exe",
            "gnfinder",
        ]
        
        for path in gnfinder_paths:
            try:
                # Check if file exists first
                if not os.path.exists(path):
                    continue

                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.gnfinder_path = path
                    self.gnfinder_available = True
                    version = result.stdout.strip() if result.stdout else "unknown"
                    print(f"  ✓ GNfinder found at: {path} (version: {version})")
                    return  # This return is fine - exits the method
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
                continue
        
        # If we get here, gnfinder wasn't found
        print("  ⚠️ GNfinder not found. This is optional - other models will still work.")
        print("     To install GNfinder:")
        print("     1. Download from: https://github.com/gnames/gnfinder/releases")
        print("     2. Place gnfinder.exe in C:\\bin or add to PATH")
        print("     3. Or place it in your project folder")


    def load_spacy(self):
        """Load spaCy model on demand."""
        if self.models_loaded["spacy"]:
            return
        
        try:
            print("  Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            self.models_loaded["spacy"] = True
            print("  ✓ spaCy model loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading spaCy: {e}")
            print("  spaCy will be disabled for this session.")


    def load_gliner2(self):
        """Load GLiNER2 model on demand."""
        if self.models_loaded["gliner2"]:
            return
        
        try:
            print("  Loading GLiNER2 model...")
            self.gliner2 = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
            self.models_loaded["gliner2"] = True
            print("  ✓ GLiNER2 model loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading GLiNER2: {e}")
            print("  GLiNER2 will be disabled for this session.")


    def load_gliner(self):
        """Load updated GLiNER model on demand."""
        if self.models_loaded["gliner"]:
            return
        
        try:
            print("  Loading updated GLiNER model...")
            self.gliner = GLiNER.from_pretrained("gliner-community/gliner_medium-v2.5",  
                load_tokenizer=True
            )
            self.models_loaded["gliner"] = True
            print("  ✓ Updated GLiNER model loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading updated GLiNER: {e}")
            print("  Updated GLiNER will be disabled for this session.")


    def load_gnfinder(self):
        """Load/verify GNfinder is available."""
        if self.models_loaded["gnfinder"]:
            return
        
        if not self.gnfinder_available or self.gnfinder_path is None:
            print("  ⚠️ GNfinder not available. Skipping.")
            return
        
        try:
            # Verify gnfinder is working
            result = subprocess.run(
                [self.gnfinder_path, "-V"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.models_loaded["gnfinder"] = True
                print("  ✓ GNfinder ready")
            else:
                print(f"  ✗ GNfinder verification failed: {result.stderr}")
        except Exception as e:
            print(f"  ✗ Error loading GNfinder: {e}")

    


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
        self.load_spacy()

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

    def process_gliner2(self, text, min_confidence=0.4):
        """Extract custom entities using GLiNER2 zero-shot."""
        self.load_gliner2()
        
        if self.gliner2 is None or not text:
            return []

        try:
            labels = ["species", "habitat", "location", "date", "person", "organism"]
            
            if len(text) < 100:
                threshold = 0.5
            else:
                threshold = 0.3
            
            predictions = self.gliner2.extract_entities(text, labels, threshold=threshold)

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

            
            filtered = []
            for entity in entities:
                # Skip very short entities (likely false positives)
                if entity["label"] == "SPECIES" and len(entity["text"]) < 3:
                    continue
                
                # Skip entities that are just numbers or codes
                if entity["label"] == "SPECIES" and entity["text"].isdigit():
                    continue
                
                # Skip entities with high number-to-letter ratio
                if entity["label"] == "SPECIES":
                    letters = sum(c.isalpha() for c in entity["text"])
                    numbers = sum(c.isdigit() for c in entity["text"])
                    if letters == 0 or numbers > letters * 2:
                        continue
                
                filtered.append(entity)
        
            print(f"  GLiNER2 found {len(entities)} entities, filtered to {len(filtered)} after post-processing")
            return filtered

        except Exception as e:
            print(f"  ✗ Error during GLiNER2 processing: {e}")
            return []

        
    def process_gliner(self, text, threshold=0.5):
        """Process text using the GLiNER model."""
        self.load_gliner()

        if self.gliner is None or not text:
            return []

        try:
            labels = [
                "species", "habitat", "location", "date", "person", "organism"
            ]

            # Dynamic threshold based on text length
            if len(text) < 100:
                threshold = 0.6
            elif len(text) < 500:
                threshold = 0.5
            else:
                threshold = 0.4

            predictions = self.gliner.predict_entities(text, labels, threshold=threshold)

            entities = []

            if isinstance(predictions, list):
                for pred in predictions:
                    if isinstance(pred, dict):
                        entity_text = pred.get('text', '')
                        label = pred.get('label', '').upper()
                        score = pred.get('score', 0.5)
                        
                        if entity_text:
                            # Find position in original text
                            start_idx = text.lower().find(entity_text.lower())
                            
                            if start_idx != -1:
                                entities.append({
                                    "text": entity_text,
                                    "label": standardise_label(label),
                                    "score": score,
                                    "source": "GLiNER",
                                    "start": start_idx,
                                    "end": start_idx + len(entity_text)
                                })

            # Handle dict format 
            elif isinstance(predictions, dict):
                if "entities" in predictions:
                    for pred in predictions["entities"]:
                        entity_text = pred.get('text', '')
                        label = pred.get('label', '').upper()
                        score = pred.get('score', 0.5)
                        
                        if entity_text:
                            start_idx = text.lower().find(entity_text.lower())
                            if start_idx != -1:
                                entities.append({
                                    "text": entity_text,
                                    "label": standardise_label(label),
                                    "score": score,
                                    "source": "GLiNER",
                                    "start": start_idx,
                                    "end": start_idx + len(entity_text)
                                })
                else:
                    for label, text_list in predictions.items():
                        if isinstance(text_list, list):
                            for entity_text in text_list:
                                if entity_text:
                                    start_idx = text.lower().find(entity_text.lower())
                                    if start_idx != -1:
                                        entities.append({
                                            "text": entity_text,
                                            "label": standardise_label(label.upper()),
                                            "score": 0.5,  # No confidence in this format
                                            "source": "GLiNER",
                                            "start": start_idx,
                                            "end": start_idx + len(entity_text)
                                        })

            filtered_entities = []
            for entity in entities:
                entity_text = entity["text"]
                label = entity["label"]
                
                if len(entity_text) < 2:
                    continue
                
                if entity_text.strip().isdigit():
                    continue
                
                if label == "SPECIES":
                    letters = sum(c.isalpha() for c in entity_text)
                    numbers = sum(c.isdigit() for c in entity_text)
                    if letters == 0 or numbers > letters * 2:
                        continue
                
                if label == "LOCATION":
                    common_words = ["here", "there", "where", "this", "that", "then", "now"]
                    if entity_text.lower() in common_words:
                        continue
                
                if label == "SPECIES":
                    if len(entity_text) == 1:
                        continue
                    
                    if " " in entity_text:
                        parts = entity_text.split()
                        if len(parts) >= 2:
                            # Check if it looks like a scientific name
                            # First part should be capitalized (genus)
                            if not parts[0][0].isupper():
                                # Could still be "sp. nov." format
                                if "sp." not in entity_text.lower():
                                    continue
                    else:
                        # Single word - should be capitalized if it's a genus
                        if not entity_text[0].isupper():
                            continue
                
                # Validate dates
                if label == "DATE":
                    if len(entity_text) == 1 and entity_text.isdigit():
                        continue
                
                # If all filters passed, keep the entity
                filtered_entities.append(entity)

            print(f"  GLiNER found {len(entities)} entities, kept {len(filtered_entities)} after filtering")
            return filtered_entities

        except Exception as e:
            print(f"  ✗ Error during GLiNER processing: {e}")
            import traceback
            print(traceback.format_exc())
            return []


    

    def process_gnfinder(self, text, verify=False, language="eng"):
        """Process text using GNfinder with correct CSV parsing."""
        self.load_gnfinder()
        
        if not self.models_loaded["gnfinder"] or not text:
            return []

        try:
            import subprocess
            import tempfile
            import os
            import csv
            import io
            
            # Create temp file with the text
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.txt', 
                delete=False, 
                encoding='utf-8'
            ) as f:
                f.write(text)
                temp_path = f.name
            
            # Build command - use CSV format explicitly
            cmd = [
                self.gnfinder_path,
                "-U",  # Plain text
                "-f", "csv",  # Explicitly request CSV
                temp_path
            ]
            
            # Optional verification (requires internet)
            if verify:
                cmd.append("-v")
            
            # Run gnfinder
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            if result.returncode != 0:
                if result.stderr:
                    print(f"  ✗ GNfinder error: {result.stderr[:200]}")
                return []
            
            if not result.stdout or result.stdout.strip() == "":
                return []
            
            # Parse CSV output
            entities = []
            
            # Use csv.reader for proper CSV parsing
            csv_reader = csv.reader(io.StringIO(result.stdout))
            
            # Get header
            try:
                header = next(csv_reader)
            except StopIteration:
                return []
            
            # Find column indices
            try:
                name_idx = header.index('Name')
                verbatim_idx = header.index('Verbatim')
                start_idx_col = header.index('Start')
                end_idx_col = header.index('End')
                odds_idx = header.index('OddsLog10')
            except ValueError:
                print(f"  ✗ Unexpected CSV header: {header}")
                return []
            
            # Process each row
            for row in csv_reader:
                if len(row) <= max(name_idx, verbatim_idx, start_idx_col, end_idx_col, odds_idx):
                    continue
                
                # Get entity text (prefer Name, fallback to Verbatim)
                entity_text = row[name_idx].strip() if name_idx < len(row) else ""
                if not entity_text:
                    entity_text = row[verbatim_idx].strip() if verbatim_idx < len(row) else ""
                
                if not entity_text:
                    continue
                
                # Get confidence score (OddsLog10)
                score = 0.5
                odds_value = None
                try:
                    if odds_idx < len(row) and row[odds_idx] and row[odds_idx].strip():
                        odds_value = float(row[odds_idx])
                        # Convert log odds to probability
                        score = 1 / (1 + 10 ** (-odds_value))
                    else:
                        score = 0.5  # Default if empty
                except (ValueError, IndexError):
                    score = 0.5  # Default if conversion fails
                
                # Get position
                start_pos = -1
                end_pos = -1
                try:
                    if start_idx_col < len(row):
                        start_pos = int(row[start_idx_col])
                    if end_idx_col < len(row):
                        end_pos = int(row[end_idx_col])
                except (ValueError, IndexError):
                    pass
                
                # If positions are available, use them
                if start_pos != -1 and end_pos != -1:
                    # Double-check with text.find
                    found_pos = text.find(entity_text)
                    if found_pos != -1:
                        entities.append({
                            "text": entity_text,
                            "label": "SPECIES",
                            "source": "GNfinder",
                            "start": found_pos,
                            "end": found_pos + len(entity_text),
                            "score": score,
                            "odds_log10": float(row[odds_idx]) if odds_idx < len(row) else None
                        })
                    else:
                        # Try case-insensitive search
                        for i in range(len(text) - len(entity_text) + 1):
                            if text[i:i+len(entity_text)].lower() == entity_text.lower():
                                entities.append({
                                    "text": entity_text,
                                    "label": "SPECIES",
                                    "source": "GNfinder",
                                    "start": i,
                                    "end": i + len(entity_text),
                                    "score": score,
                                    "odds_log10": float(row[odds_idx]) if odds_idx < len(row) else None
                                })
                                break
                else:
                    # Fallback to text.find
                    start_pos = text.lower().find(entity_text.lower())
                    if start_pos != -1:
                        entities.append({
                            "text": entity_text,
                            "label": "SPECIES",
                            "source": "GNfinder",
                            "start": start_pos,
                            "end": start_pos + len(entity_text),
                            "score": score,
                            "odds_log10": float(row[odds_idx]) if odds_idx < len(row) else None
                        })
            
            print(f"  GNfinder found {len(entities)} species names")
            return entities
            
        except subprocess.TimeoutExpired:
            print("  ✗ GNfinder timed out")
            return []
        except Exception as e:
            print(f"  ✗ Error during GNfinder processing: {e}")
            return []
    


    def process_text(self, text, method="hybrid"):
        """
        Process text with selected NER method.

        Args:
            text (str): Input text
            method (str): One of:
                - "spacy_only" / "spacy only"
                - "gliner2_only" / "gliner2 only"
                - "gliner_only" / "gliner only"
                - "gnfinder_only" / "gnfinder only"  # NEW
                - "hybrid" (spaCy + GLiNER2 - default)
                - "hybrid_g" (spaCy + GLiNER)
                - "hybrid_gn" (spaCy + GLiNER + GNfinder)  # NEW

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
                "gliner2_count": 0,
                "gliner_count": 0,
                "gnfinder_count": 0,
                "error": "Invalid input"
            }

        print("\n" + "=" * 60)
        print(f"Processing Text (Method: {method})")
        print("=" * 60)
        print(f"Text length: {len(text)} characters\n")

        try:
            spacy_entities = []
            gliner2_entities = []
            gliner_entities = []
            gnfinder_entities = []

            method_lower = method.lower().strip()

            # spaCy only
            if method_lower == "spacy only" or method_lower == "spacy_only":
                print("Running spaCy only...")
                spacy_entities = self.process_spacy(text)
                print(f"  spaCy found {len(spacy_entities)} entities")
                all_entities = spacy_entities

            # GLiNER2 only
            elif method_lower == "gliner2 only" or method_lower == "gliner2_only":
                print("Running GLiNER2 only...")
                gliner2_entities = self.process_gliner2(text)
                print(f"  GLiNER2 found {len(gliner2_entities)} entities")
                all_entities = gliner2_entities

            # GLiNER only
            elif method_lower == "gliner only" or method_lower == "gliner_only":
                print("Running updated GLiNER only...")
                gliner_entities = self.process_gliner(text)
                print(f"  Updated GLiNER found {len(gliner_entities)} entities")
                all_entities = gliner_entities

            # GNfinder only
            elif method_lower == "gnfinder only" or method_lower == "gnfinder_only":
                print("Running GNfinder only...")
                gnfinder_entities = self.process_gnfinder(text)
                print(f"  GNfinder found {len(gnfinder_entities)} entities")
                all_entities = gnfinder_entities

            # Hybrid: spaCy + GLiNER
            elif method_lower == "hybrid g" or method_lower == "hybrid_g":
                print("Running spaCy + updated GLiNER hybrid...")
                spacy_entities = self.process_spacy(text)
                print(f"  spaCy found {len(spacy_entities)} entities")
                gliner_entities = self.process_gliner(text)
                print(f"  Updated GLiNER found {len(gliner_entities)} entities")
                all_entities = spacy_entities + gliner_entities

            # Hybrid: spaCy + GLiNER + GNfinder
            elif method_lower == "hybrid gn" or method_lower == "hybrid_gn" or method_lower == "hybrid_gnfinder":
                print("Running spaCy + GLiNER + GNfinder hybrid...")
                spacy_entities = self.process_spacy(text)
                gliner_entities = self.process_gliner(text)
                gnfinder_entities = self.process_gnfinder(text)
                print(f"  spaCy: {len(spacy_entities)}, GLiNER: {len(gliner_entities)}, GNfinder: {len(gnfinder_entities)}")
                all_entities = spacy_entities + gliner_entities + gnfinder_entities

            # Default: hybrid (spaCy + GLiNER2)
            else:
                print("Running spaCy + GLiNER2 hybrid...")
                spacy_entities = self.process_spacy(text)
                print(f"  spaCy found {len(spacy_entities)} entities")
                gliner2_entities = self.process_gliner2(text)
                print(f"  GLiNER2 found {len(gliner2_entities)} entities")
                all_entities = spacy_entities + gliner2_entities

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
                "gliner2_count": len(gliner2_entities),
                "gliner_count": len(gliner_entities),
                "gnfinder_count": len(gnfinder_entities),
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
                "gliner2_count": 0,
                "gliner_count": 0,
                "gnfinder_count": 0,
                "error": str(e)
            }


if __name__ == "__main__":
    print("-" * 60)
    print("NER Processor Test")
    print("-" * 60)