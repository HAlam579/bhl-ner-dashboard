import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ner_processor import NERProcessor


def test_normalisation():
    
    processor = NERProcessor()
    
    print("="*70)
    print("TESTING TEXT NORMALISATION")
    print("="*70)
    
    test_cases = [

        {
            "name": "Multiple spaces",
            "input": "Panda    lives    in    China",
            "expected": "Panda lives in China"
        },
        {
            "name": "Newlines",
            "input": "Panda\nlives\nin\nChina",
            "expected": "Panda lives in China"
        },
        {
            "name": "Carriage returns",
            "input": "Panda\rlives\rin\rChina",
            "expected": "Panda lives in China"
        },
        {
            "name": "Mixed whitespace",
            "input": "Panda  \n  lives  \r\n  in  \t  China",
            "expected": "Panda lives in China"
        },
        {
            "name": "Leading/trailing spaces",
            "input": "  Panda lives in China  ",
            "expected": "Panda lives in China"
        },
        {
            "name": "Multiple newlines",
            "input": "Panda\n\n\nlives\n\n\nin\n\n\nChina",
            "expected": "Panda lives in China"
        },
        {
            "name": "Tabs and spaces",
            "input": "Panda\t\tlives\t\tin\t\tChina",
            "expected": "Panda lives in China"
        },
        {
            "name": "Complex mixed",
            "input": "  \n  Panda  \t  lives  \r\n  in  \t  China  \n  ",
            "expected": "Panda lives in China"
        },
        {
            "name": "Empty string",
            "input": "",
            "expected": ""
        },
        {
            "name": "Only whitespace",
            "input": "   \n  \t  \r\n  ",
            "expected": ""
        },
        {
            "name": "None input",
            "input": None,
            "expected": ""
        },
        {
            "name": "Non-string input",
            "input": 12345,
            "expected": ""
        },
        {
            "name": "Sentence with internal spaces",
            "input": "The  giant  panda  Ailuropoda  melanoleuca  is  a  bear",
            "expected": "The giant panda Ailuropoda melanoleuca is a bear"
        },
        {
            "name": "BHL-style text with newlines",
            "input": "Page 1\n\nSpecies: Ailuropoda melanoleuca\nLocation: China\nDate: 1869",
            "expected": "Page 1 Species: Ailuropoda melanoleuca Location: China Date: 1869"
        },
        {
            "name": "Multiple sentences with spacing",
            "input": "Pandas  eat  bamboo.  They  live  in  China.  They  are  endangered.",
            "expected": "Pandas eat bamboo. They live in China. They are endangered."
        },
    ]
    
    passed = 0
    failed = 0
    
    print(f"\n{'Test Case':<40} {'Status':<10} {'Result'}")
    print("-"*70)
    
    for test in test_cases:
        result = processor.normalise_text(test["input"])
        
        # Check if result matches expected
        if result == test["expected"]:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            
            # Show debug info for failures
            print(f"\n  Input:    '{test['input']}'")
            print(f"  Expected: '{test['expected']}'")
            print(f"  Got:      '{result}'")
        
        # Show preview for long texts
        preview = result[:30] + "..." if len(result) > 30 else result
        print(f"{test['name']:<40} {status:<10} '{preview}'")
    
    # Test with actual NER processing
    print("\n" + "="*70)
    print("TESTING NORMALISATION WITH NER PROCESSING")
    print("="*70)
    
    messy_text = """
    The  giant  panda  Ailuropoda  melanoleuca
    
    is  a  bear  native  to  China.
    
    It  was  first  described  by  Armand  David  in  1869.
    """
    
    print(f"\nMessy input text:\n{messy_text}")
    
    # Process with normalisation
    results = processor.process_text(messy_text, method="hybrid")
    
    print(f"\nProcessed text (normalised):")
    print("-"*40)


    print(f"Entities found: {results['total']}")
    
    if results["entities"]:
        print("\nEntities detected:")
        for entity in results["entities"]:
            print(f"  {entity['text']} -> {entity['label']}")
    

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n✅ All normalisation tests passed!")
    else:
        print(f"\n⚠️ {failed} tests failed. Check the output above.")
    
    print("="*70)


def test_normalisation_edge_cases():
    
    processor = NERProcessor()
    
    print("\n" + "="*70)
    print("EDGE CASE NORMALISATION TESTS")
    print("="*70)
    
    edge_cases = [
        ("Very long word", "A" * 1000, "A" * 1000),
        ("Special characters", "Panda!@#$%^&*()", "Panda!@#$%^&*()"),
        ("Unicode", "Panda 🐼 China 🇨🇳", "Panda 🐼 China 🇨🇳"),
        ("Numbers", "1869 1758 1835", "1869 1758 1835"),
        ("Mixed case", "pAnDa cHiNa", "pAnDa cHiNa"),
    ]
    
    passed = 0
    failed = 0
    
    print(f"\n{'Test Case':<30} {'Status':<10} {'Result Preview'}")
    print("-"*70)
    
    for name, input_text, expected in edge_cases:
        result = processor.normalise_text(input_text)
        
        if result == expected:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        preview = result[:30] + "..." if len(result) > 30 else result
        print(f"{name:<30} {status:<10} '{preview}'")
    
    print("\n" + "="*70)
    if failed == 0:
        print("✅ All edge case tests passed!")
    else:
        print(f"⚠️ {failed} edge case tests failed.")
    print("="*70)


if __name__ == "__main__":
    test_normalisation()
    test_normalisation_edge_cases()