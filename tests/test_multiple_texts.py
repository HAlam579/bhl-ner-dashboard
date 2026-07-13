"""
Test the NER Processor on Multiple Texts with GNfinder.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ner_processor import NERProcessor


def run_test(processor, text, test_name, method="hybrid_gn"):
    """Run a single test and print formatted results."""
    print("\n" + "=" * 70)
    print(f"TEST: {test_name} (Method: {method})")
    print("=" * 70)
    
    preview = text[:200].replace('\n', ' ')
    print(f"Text preview: {preview}...\n")
    
    results = processor.process_text(text, method=method)
    
    print(f"Found {results['total']} entities")
    print(f"  spaCy: {results['spacy_count']}, GLiNER2: {results['gliner2_count']}, GLiNER: {results['gliner_count']}, GNfinder: {results['gnfinder_count']}")
    
    if results["entities"]:
        print("\nEntities:")
        for entity in results["entities"]:
            score_info = f" (score: {entity.get('score', 'N/A'):.4f})" if entity.get('score') and isinstance(entity.get('score'), (int, float)) else ""
            print(f"  - '{entity['text']}' -> {entity['label']} ({entity['source']}){score_info}")
    
    print(f"\nEntity counts by type: {results['counts']}")
    
    return {
        "name": test_name,
        "total": results["total"],
        "spacy_count": results["spacy_count"],
        "gliner2_count": results["gliner2_count"],
        "gliner_count": results["gliner_count"],
        "gnfinder_count": results["gnfinder_count"],
        "counts": results["counts"]
    }


def main():
    print("=" * 70)
    print("Testing NER Processor on Multiple Texts (With GNfinder)")
    print("=" * 70)
    
    print("\nLoading NER models...")
    processor = NERProcessor()
    
    texts = [
        ("1: Panda", """
        The giant panda Ailuropoda melanoleuca is a bear native to China.
        It was first described by French missionary Armand David in 1869.
        The panda lives in the mountains of Sichuan province.
        Its habitat is threatened by deforestation.
        """),
        
        ("2: Whale", """
        The blue whale Balaenoptera musculus is the largest animal on Earth.
        It can grow up to 30 metres in length and weigh over 150 tonnes.
        Blue whales are found in all oceans except the Arctic.
        """),
        
        ("3: Darwin", """
        Charles Darwin published On the Origin of Species in 1859.
        He visited the Galapagos Islands in 1835 on HMS Beagle.
        """),
        
        ("4: Butterfly", """
        The monarch butterfly Danaus plexippus is known for its migration.
        It travels from North America to Mexico each winter.
        """),
        
        ("5: Fungi", """
        Amanita muscaria is a poisonous mushroom found in the UK.
        It grows in forests with birch and pine trees.
        """),
    ]
    
    results_summary = []
    for name, text in texts:
        results_summary.append(run_test(processor, text, name, "hybrid_gn"))
    
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"{'Test Name':<20} {'Total':<8} {'spaCy':<8} {'GLiNER2':<8} {'GLiNER':<8} {'GNfinder':<8}")
    print("-" * 70)
    
    for summary in results_summary:
        print(f"{summary['name']:<20} {summary['total']:<8} {summary['spacy_count']:<8} {summary['gliner2_count']:<8} {summary['gliner_count']:<8} {summary['gnfinder_count']:<8}")
    
    print("-" * 70)
    
    avg_total = sum(s["total"] for s in results_summary) / len(results_summary)
    avg_spacy = sum(s["spacy_count"] for s in results_summary) / len(results_summary)
    avg_gliner2 = sum(s["gliner2_count"] for s in results_summary) / len(results_summary)
    avg_gliner = sum(s["gliner_count"] for s in results_summary) / len(results_summary)
    avg_gnfinder = sum(s["gnfinder_count"] for s in results_summary) / len(results_summary)
    
    print(f"\n{'Average':<20} {avg_total:<8.1f} {avg_spacy:<8.1f} {avg_gliner2:<8.1f} {avg_gliner:<8.1f} {avg_gnfinder:<8.1f}")


if __name__ == "__main__":
    main()