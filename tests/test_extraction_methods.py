"""
Test script to verify all three extraction methods.
Run with: py tests/test_extraction_methods.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner_processor import NERProcessor

print("=" * 60)
print("Testing Extraction Methods")
print("=" * 60)

# Test text
test_text = "The giant panda Ailuropoda melanoleuca is a bear native to China."

print(f"\nTest text: {test_text}")

# Create processor
print("\nLoading NER processor...")
processor = NERProcessor()

# Test 1: spaCy Only
print("\n" + "=" * 60)
print("TEST 1: spaCy Only")
print("=" * 60)
results_spacy = processor.process_text(test_text, method="spacy_only")
print(f"Found {results_spacy['total']} entities")
for entity in results_spacy["entities"]:
    print(f"  - {entity['text']} -> {entity['label']} ({entity['source']})")

# Test 2: GLiNER2 Only
print("\n" + "=" * 60)
print("TEST 2: GLiNER2 Only")
print("=" * 60)
try:
    results_gliner = processor.process_text(test_text, method="gliner2_only")
    print(f"Found {results_gliner['total']} entities")
    for entity in results_gliner["entities"]:
        score_info = f" (score: {entity.get('score', 'N/A'):.2f})" if entity.get('score') else ""
        print(f"  - {entity['text']} -> {entity['label']} ({entity['source']}){score_info}")
except Exception as e:
    print(f"Error in GLiNER2 extraction: {e}")

# Test 3: Hybrid
print("\n" + "=" * 60)
print("TEST 3: Hybrid (spaCy + GLiNER2)")
print("=" * 60)
results_hybrid = processor.process_text(test_text, method="hybrid")
print(f"Found {results_hybrid['total']} entities (spaCy: {results_hybrid['spacy_count']}, GLiNER2: {results_hybrid['gliner_count']})")
for entity in results_hybrid["entities"]:
    score_info = f" (score: {entity.get('score', 'N/A'):.2f})" if entity.get('score') else ""
    print(f"  - {entity['text']} -> {entity['label']} ({entity['source']}){score_info}")

print("\n" + "=" * 60)
print("Test complete.")
print("=" * 60)