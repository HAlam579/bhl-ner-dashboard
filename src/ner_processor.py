"""
NER Processor for BHL Dashboard.
Extracts entities using spaCy and GLiNER with error handling.
"""

import spacy
from gliner import GLiNER
from collections import Counter
import sys
import traceback

class NERProcessor:
    """
    Process text using spaCy and GLiNER.
    Handles errors - script does not crash.
    """
    
    def __init__(self):
        """
        Initialize both NER models with error handling.
        
        If a model fails to load, the script will continue
        with the other model if possible.
        """
        print("Loading NER models...")
        
        self.nlp = None
        try:
            print("  Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            print("  ✓ spaCy model loaded successfully")
        except OSError as e:
            print(f"  ✗ Error loading spaCy model: {e}")
            print("  Please install the model with: py -m spacy download en_core_web_sm")
        except Exception as e:
            print(f"  ✗ Unexpected error loading spaCy: {e}")
            print("  spaCy will be disabled for this session.")
        

        self.gliner = None
        try:
            print("  Loading GLiNER model...")
            self.gliner = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
            print("  ✓ GLiNER model loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading GLiNER model: {e}")
            print("  GLiNER will be disabled for this session.")
        
        # Check if any models loaded successfully
        if self.nlp is None and self.gliner is None:
            print("\n⚠ WARNING: Neither spaCy nor GLiNER loaded successfully.")
            print("The processor will not work. Please check your installation.")
        elif self.nlp is None:
            print("\n⚠ WARNING: spaCy is disabled. Only GLiNER will be used.")
        elif self.gliner is None:
            print("\n⚠ WARNING: GLiNER is disabled. Only spaCy will be used.")
        else:
            print("\n✓ Both models loaded successfully.")
    
    def process_spacy(self, text):
        """
        Extract standard entities using spaCy with error handling.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of entities found, or empty list if error
        """
        # If spaCy is not available, return empty list
        if self.nlp is None:
            print("  spaCy is not available. Skipping.")
            return []
        
        # If text is empty or invalid, return empty list
        if not text or not isinstance(text, str):
            print("  Invalid input text. Skipping spaCy.")
            return []
        
        try:
            # Process the text with spaCy
            doc = self.nlp(text)
            entities = []
            
            labels = ["PERSON", "GPE", "DATE", "ORG"]
            
            for ent in doc.ents:
                if ent.label_ in labels:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "source": "spaCy"
                    })
            
            return entities
            
        except Exception as e:
            # Catch any error during spaCy processing
            print(f"  ✗ Error during spaCy processing: {e}")
            return []
    
    def process_gliner(self, text):
        """
        Extract custom entities using GLiNER zero-shot with error handling.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of entities found, or empty list if error
        """
        # If GLiNER is not available, return empty list
        if self.gliner is None:
            print("  GLiNER is not available. Skipping.")
            return []
        
        # If text is empty or invalid, return empty list
        if not text or not isinstance(text, str):
            print("  Invalid input text. Skipping GLiNER.")
            return []
        
        try:
            # Define the entity types to find
            labels = ["species", "habitat", "location", "date", "person", "organism"]
            
            predictions = self.gliner.predict_entities(
                text, 
                labels, 
                threshold=0.3
            )
            
            entities = []
            for ent in predictions:
                entities.append({
                    "text": ent["text"],
                    "label": ent["label"].upper(),
                    "score": ent["score"],
                    "source": "GLiNER"
                })
            
            return entities
            
        except Exception as e:
            # Catch any error during GLiNER processing
            print(f"  ✗ Error during GLiNER processing: {e}")
            return []
    
    def process_text(self, text):
        """
        Process text with both spaCy and GLiNER with full error handling.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Combined results, or a safe empty result on critical error
        """
        # Handle invalid input
        if not text or not isinstance(text, str):
            print("✗ Error: Input text is empty or invalid.")
            return {
                "entities": [],
                "counts": {},
                "total": 0,
                "spacy_count": 0,
                "gliner_count": 0,
                "error": "Invalid input"
            }
        
        print("\n" + "=" * 60)
        print("Processing Text")
        print("=" * 60)
        print(f"Text length: {len(text)} characters\n")
        
        # If neither model is available, return error
        if self.nlp is None and self.gliner is None:
            print("✗ Error: No NER models available.")
            return {
                "entities": [],
                "counts": {},
                "total": 0,
                "spacy_count": 0,
                "gliner_count": 0,
                "error": "No models available"
            }
        
        try:
            # Run both models 
            print("Running spaCy...")
            spacy_entities = self.process_spacy(text)
            print(f"  spaCy found {len(spacy_entities)} entities")
            
            print("Running GLiNER...")
            gliner_entities = self.process_gliner(text)
            print(f"  GLiNER found {len(gliner_entities)} entities")
            
            # Combine and deduplicate
            all_entities = spacy_entities + gliner_entities
            
            # Remove duplicates (same text + same label)
            seen = set()
            unique_entities = []
            for ent in all_entities:
                key = f"{ent['text'].lower()}|{ent['label']}"
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(ent)
            
            # Count entity types
            label_counts = Counter([ent["label"] for ent in unique_entities])
            
            return {
                "entities": unique_entities,
                "counts": dict(label_counts),
                "total": len(unique_entities),
                "spacy_count": len(spacy_entities),
                "gliner_count": len(gliner_entities)
            }
            
        except Exception as e:
            # Catch any unexpected error
            print(f"✗ Unexpected error during processing: {e}")
            print(traceback.format_exc())  # Shows where the error occurred
            return {
                "entities": [],
                "counts": {},
                "total": 0,
                "spacy_count": 0,
                "gliner_count": 0,
                "error": str(e)
            }
    
    def format_results(self, results):
        """
        Format results for display with error handling.
        
        Args:
            results (dict): Results from process_text
            
        Returns:
            str: Formatted output
        """
        # Check if there was an error
        if results.get("error"):
            output = []
            output.append("\n" + "=" * 60)
            output.append("ERROR")
            output.append("=" * 60)
            output.append(f"An error occurred: {results['error']}")
            output.append("Please check your input and try again.")
            return "\n".join(output)
        
        # If no entities were found
        if results["total"] == 0:
            output = []
            output.append("\n" + "=" * 60)
            output.append("EXTRACTED ENTITIES")
            output.append("=" * 60)
            output.append("\nNo entities found in the text.")
            output.append("\n" + "=" * 60)
            return "\n".join(output)
        
        # Normal output
        output = []
        output.append("\n" + "=" * 60)
        output.append("EXTRACTED ENTITIES")
        output.append("=" * 60)
        
        # Group by source
        spacy_ents = [e for e in results["entities"] if e["source"] == "spaCy"]
        gliner_ents = [e for e in results["entities"] if e["source"] == "GLiNER"]
        
        if spacy_ents:
            output.append("\n[spaCy] Standard Entities:")
            for e in spacy_ents:
                output.append(f"  {e['text']} -> {e['label']}")
        
        if gliner_ents:
            output.append("\n[GLiNER] Domain Entities:")
            for e in gliner_ents:
                output.append(f"  {e['text']} -> {e['label']} (score: {e['score']:.2f})")
        
        # Summary
        output.append("\n" + "=" * 60)
        output.append("SUMMARY")
        output.append("=" * 60)
        output.append(f"Total entities found: {results['total']}")
        output.append(f"  spaCy: {results['spacy_count']}")
        output.append(f"  GLiNER: {results['gliner_count']}")
        
        output.append("\nEntity type counts:")
        for label, count in results["counts"].items():
            output.append(f"  {label}: {count}")
        
        return "\n".join(output)


# --- Test the Processor ---
if __name__ == "__main__":
    print("-" * 60)
    print("NER Processor Test (with Error Handling)")
    print("-" * 60)

    # Test 1: Normal text
    print("\n--- Test 1: Normal text ---")
    sample_text = """
    The giant panda Ailuropoda melanoleuca is a bear native to China.
    It was first described by French missionary Armand David in 1869.
    """
    
    processor = NERProcessor()
    results = processor.process_text(sample_text)
    print(processor.format_results(results))

    # Test 2: Empty text
    print("\n--- Test 2: Empty text ---")
    empty_results = processor.process_text("")
    print(processor.format_results(empty_results))

    # Test 3: Very long text (simulates potential processing issues)
    print("\n--- Test 3: Long text ---")
    long_text = " ".join(["The panda lives in China." for _ in range(100)])
    long_results = processor.process_text(long_text)
    print(processor.format_results(long_results))