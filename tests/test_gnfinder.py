import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner_processor import NERProcessor

def test_gnfinder():
    print("=" * 60)
    print("GNfinder Integration Test")
    print("=" * 60)
    
    processor = NERProcessor()
    
    # Check if GNfinder is available
    if not processor.gnfinder_available:
        print("\n❌ GNfinder not found. Please install:")
        print("   1. Download from: https://github.com/gnames/gnfinder/releases")
        print("   2. Place gnfinder.exe in C:\\bin")
        print("   3. Restart your terminal")
        return
    
    print(f"\n✅ GNfinder found at: {processor.gnfinder_path}")
    
    # Test with simple text containing species names
    test_text = """Ailuropoda melanoleuca Balaenoptera musculus Danaus plexippus"""
    
    print(f"\nTest text: {test_text}")
    print("\n" + "=" * 60)
    print("Processing with GNfinder...")
    print("=" * 60)
    

    results = processor.process_gnfinder(test_text, verify=False)
    
    print(f"\nFound {len(results)} entities:")
    if results:
        for entity in results:
            print(f"  - '{entity['text']}' (score: {entity.get('score', 'N/A')})")
    else:
        print("  No entities found")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    test_gnfinder()