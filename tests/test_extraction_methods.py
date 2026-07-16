import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner_processor import NERProcessor

print("=" * 60)
print("Testing All Extraction Methods")
print("=" * 60)


test_text = """The giant panda Ailuropoda melanoleuca is a bear native to China.
It was first described by French missionary Armand David in 1869.
The habitat includes bamboo forests in Sichuan province."""

print(f"\nTest text: {test_text}")


print("\nLoading NER processor...")
processor = NERProcessor()


methods = [
    ("spacy_only", "spaCy Only"),
    ("gliner2_only", "GLiNER2 Only"),
    ("gliner_only", "GLiNER Only"),
    ("gnfinder_only", "GNfinder Only"),
    # ("taxonerd_only", "TaxoNERD Only"),
    ("hybrid", "Hybrid (spaCy + GLiNER2)"),
    ("hybrid_g", "Hybrid (spaCy + GLiNER)"),
    ("hybrid_gn", "Hybrid (spaCy + GLiNER + GNfinder)"),
    # ("hybrid_tx", "Hybrid (spaCy + GLiNER + TaxoNERD)"),
]

for method_name, display_name in methods:
    print("\n" + "=" * 60)
    print(f"TEST: {display_name}")
    print("=" * 60)
    
    try:
        results = processor.process_text(test_text, method=method_name)
        print(f"Found {results['total']} entities")
        print(f"  spaCy: {results['spacy_count']}, GLiNER2: {results['gliner2_count']}, GLiNER: {results['gliner_count']}, GNfinder: {results['gnfinder_count']}")
        
        if results["entities"]:
            print("\nEntities:")
            for entity in results["entities"]:
                score_info = f" (score: {entity.get('score', 'N/A'):.2f})" if entity.get('score') and isinstance(entity['score'], (int, float)) else ""
                print(f"  - '{entity['text']}' -> {entity['label']} ({entity['source']}){score_info}")
        else:
            print("  No entities found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("All tests complete.")
print("=" * 60)