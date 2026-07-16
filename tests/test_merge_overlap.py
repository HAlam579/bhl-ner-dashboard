import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner_processor import NERProcessor


def test_merge_overlapping():
    """Test that overlapping entities are merged correctly."""
    
    processor = NERProcessor()
    
    # Test case 1: Species should beat PERSON
    entities = [
        {"text": "Ailuropoda melanoleuca", "label": "SPECIES", "source": "GLiNER2", "start": 16, "end": 38},
        {"text": "Ailuropoda", "label": "PERSON", "source": "spaCy", "start": 16, "end": 26},
    ]
    
    merged = processor._merge_overlapping_entities(entities)
    
    print("Test 1: Species vs PERSON")
    print(f"  Before: {len(entities)} entities")
    print(f"  After: {len(merged)} entities")
    
    # Should keep the SPECIES entity
    assert len(merged) == 1, f"Expected 1, got {len(merged)}"
    assert merged[0]["label"] == "SPECIES", f"Expected SPECIES, got {merged[0]['label']}"
    assert merged[0]["text"] == "Ailuropoda melanoleuca"
    print("  ✅ PASS")
    
    # Test case 2: No overlap - keep both
    entities = [
        {"text": "China", "label": "LOCATION", "source": "spaCy", "start": 48, "end": 53},
        {"text": "Armand David", "label": "PERSON", "source": "spaCy", "start": 68, "end": 80},
    ]
    
    merged = processor._merge_overlapping_entities(entities)
    
    print("\nTest 2: No overlap")
    print(f"  Before: {len(entities)} entities")
    print(f"  After: {len(merged)} entities")
    
    assert len(merged) == 2, f"Expected 2, got {len(merged)}"
    print("  ✅ PASS")
    
    # Test case 3: Same label, longer wins
    entities = [
        {"text": "Charles Darwin", "label": "PERSON", "source": "spaCy", "start": 0, "end": 14},
        {"text": "Darwin", "label": "PERSON", "source": "spaCy", "start": 9, "end": 14},
    ]
    
    merged = processor._merge_overlapping_entities(entities)
    
    print("\nTest 3: Longer text wins")
    print(f"  Before: {len(entities)} entities")
    print(f"  After: {len(merged)} entities")
    
    assert len(merged) == 1, f"Expected 1, got {len(merged)}"
    assert merged[0]["text"] == "Charles Darwin", f"Expected 'Charles Darwin', got '{merged[0]['text']}'"
    print("  ✅ PASS")
    
    print("\n✅ All merge tests passed!")


if __name__ == "__main__":
    test_merge_overlapping()