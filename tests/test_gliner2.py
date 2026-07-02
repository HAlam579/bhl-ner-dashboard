"""
Test script to verify GLiNER2 installation and functionality.
Run with: py tests/test_gliner2.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("Testing GLiNER2 Installation")
print("=" * 60)

# Test 1: Import GLiNER2
print("\n1. Testing import...")
try:
    from gliner2 import GLiNER2
    print("   ✓ GLiNER2 imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import GLiNER2: {e}")
    print("   Please run: py -m pip install gliner2[local]")
    exit()

# Test 2: Load the model
print("\n2. Loading GLiNER2 model...")
try:
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    exit()

# Test 3: Extract entities
print("\n3. Testing entity extraction...")
test_text = "The giant panda Ailuropoda melanoleuca lives in China."
labels = ["species", "location", "person", "date"]

try:
    predictions = model.extract_entities(test_text, labels, threshold=0.3)
    print(f"   Found {len(predictions)} entities:")
    for entity in predictions:
        print(f"     - {entity['text']} -> {entity['label']} (score: {entity.get('score', 'N/A'):.2f})")
    
    # Check if species was found
    species_found = any(e["label"] == "species" for e in predictions)
    if species_found:
        print("\n   ✅ GLiNER2 found a species!")
    else:
        print("\n   ⚠️ GLiNER2 did NOT find a species. Try lowering the threshold.")
        print("   You can modify the threshold in the code above.")
        
except Exception as e:
    print(f"   ✗ Error during extraction: {e}")

print("\n" + "=" * 60)
print("Test complete.")
print("=" * 60)