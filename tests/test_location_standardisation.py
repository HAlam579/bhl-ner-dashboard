import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner_processor import NERProcessor

processor = NERProcessor()

# Test text with both GPE and LOC
test_text = "China is in Asia. Beijing is the capital."

results = processor.process_text(test_text, method="spacy_only")

print("Entities found:")
for entity in results["entities"]:
    print(f"  {entity['text']} -> {entity['label']} (original: {entity.get('original_label', 'N/A')})")