"""
Simplified Extra Long Sentences Test.
Tests only 4 key lengths to verify performance.
Run with: py tests/test_extra_long_sentences_simple.py
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner_processor import NERProcessor


def test_extra_long():
    """Test a few key lengths only."""
    
    print("\n" + "="*60)
    print("EXTRA LONG SENTENCE TEST (Simplified)")
    print("="*60)
    
    processor = NERProcessor()
    
    test_sentences = {
        "Short (50 words)": "The giant panda Ailuropoda melanoleuca is a bear species native to China, first described by Armand David in 1869. " * 2,
        
        "Medium (100 words)": "Charles Darwin visited the Galapagos Islands in 1835 and studied finches. The blue whale Balaenoptera musculus is the largest animal. " * 5,
        
        "Long (200 words)": "Amanita muscaria grows in birch forests of the UK and contains psychoactive compounds. The monarch butterfly migrates from North America to Mexico. " * 10,
        
        "Very Long (500 words)": "Scientists study species like pandas, whales, butterflies, and mushrooms in locations worldwide. " * 25,
    }
    
    print(f"\n{'Length':<20} {'Words':<10} {'Entities':<12} {'Time (s)':<10}")
    print("-"*60)
    
    for name, text in test_sentences.items():
        start = time.time()
        result = processor.process_text(text, method="hybrid")
        elapsed = time.time() - start
        
        words = len(text.split())
        
        print(f"{name:<20} {words:<10} {result['total']:<12} {elapsed:<10.2f}")
        
        # Show entity types found
        if result["entities"]:
            types = set(e["label"] for e in result["entities"])
            print(f"  Types: {', '.join(types)}")
    
    print("\n" + "="*60)
    print("✓ Test complete!")
    print("="*60)


if __name__ == "__main__":
    test_extra_long()