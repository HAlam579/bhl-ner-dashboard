"""
Test the NER Processor on Multiple Texts.
Tests the processor on 5 different sample texts covering various topics.
"""

import sys
import os

# Add the src folder to the path so we can import the processor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ner_processor import NERProcessor


def run_test(processor, text, test_name):
    """
    Run a single test and print formatted results.
    
    Args:
        processor: The NERProcessor instance
        text (str): The text to process
        test_name (str): Name of the test for display
    """
    print("\n" + "=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)
    
    # Show a preview of the text
    preview = text[:200].replace('\n', ' ')
    print(f"Text preview: {preview}...\n")
    
    # Process the text
    results = processor.process_text(text)
    print(processor.format_results(results))
    
    # Return summary for later
    return {
        "name": test_name,
        "total": results["total"],
        "spacy_count": results["spacy_count"],
        "gliner_count": results["gliner_count"],
        "counts": results["counts"]
    }


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing NER Processor on Multiple Texts")
    print("=" * 70)
    
    # Load the processor once (models load once, reused for all tests)
    print("\nLoading NER models (this may take a moment)...")
    processor = NERProcessor()
    
    # -----------------------------------------------------------------
    # Test 1: Panda / Scientific Name
    # -----------------------------------------------------------------
    text_1 = """
    The giant panda Ailuropoda melanoleuca is a bear native to China.
    It was first described by French missionary Armand David in 1869.
    The panda lives in the mountains of Sichuan province.
    Its habitat is threatened by deforestation.
    """
    
    # -----------------------------------------------------------------
    # Test 2: Whale / Marine Biology
    # -----------------------------------------------------------------
    text_2 = """
    The blue whale Balaenoptera musculus is the largest animal on Earth.
    It can grow up to 30 metres in length and weigh over 150 tonnes.
    Blue whales are found in all oceans except the Arctic.
    They were hunted extensively in the 20th century.
    """
    
    # -----------------------------------------------------------------
    # Test 3: Darwin / Evolution / Historical Text
    # -----------------------------------------------------------------
    text_3 = """
    Charles Darwin published On the Origin of Species in 1859.
    He visited the Galapagos Islands in 1835 on HMS Beagle.
    Darwin's observations of finches and tortoises influenced his theory.
    The Galapagos Islands are part of Ecuador.
    """
    
    # -----------------------------------------------------------------
    # Test 4: Butterfly / Entomology
    # -----------------------------------------------------------------
    text_4 = """
    The monarch butterfly Danaus plexippus is known for its migration.
    It travels from North America to Mexico each winter.
    The butterfly was first described by Carl Linnaeus in 1758.
    Its habitat includes milkweed plants.
    """
    
    # -----------------------------------------------------------------
    # Test 5: Fungi / Mycology
    # -----------------------------------------------------------------
    text_5 = """
    Amanita muscaria is a poisonous mushroom found in the UK.
    It grows in forests with birch and pine trees.
    The mushroom was historically used in religious rituals.
    It contains the psychoactive compounds muscimol and ibotenic acid.
    """
    
    # -----------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------
    results_summary = []
    
    results_summary.append(run_test(processor, text_1, "1: Panda (Scientific Name)"))
    results_summary.append(run_test(processor, text_2, "2: Whale (Marine Biology)"))
    results_summary.append(run_test(processor, text_3, "3: Darwin (Evolution/Historical)"))
    results_summary.append(run_test(processor, text_4, "4: Butterfly (Entomology)"))
    results_summary.append(run_test(processor, text_5, "5: Fungi (Mycology)"))
    
    # -----------------------------------------------------------------
    # Print overall summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"{'Test Name':<30} {'Total':<8} {'spaCy':<8} {'GLiNER':<8}")
    print("-" * 70)
    
    for summary in results_summary:
        print(f"{summary['name']:<30} {summary['total']:<8} {summary['spacy_count']:<8} {summary['gliner_count']:<8}")
    
    print("-" * 70)
    
    # Calculate averages
    avg_total = sum(s["total"] for s in results_summary) / len(results_summary)
    avg_spacy = sum(s["spacy_count"] for s in results_summary) / len(results_summary)
    avg_gliner = sum(s["gliner_count"] for s in results_summary) / len(results_summary)
    
    print(f"\n{'Average':<30} {avg_total:<8.1f} {avg_spacy:<8.1f} {avg_gliner:<8.1f}")
    



if __name__ == "__main__":
    main()