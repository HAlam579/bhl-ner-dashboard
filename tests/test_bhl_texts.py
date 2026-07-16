import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fetch_bhl import get_page_text
from src.ner_processor import NERProcessor

BHL_PAGE_IDS = [
    54850898,   # Butterfly counts
    36831872,   # Bird nesting
    53548146,   # Butterfly page
    23185775,   # Mushroom   
    6970627,    # Biodiversity
    15705743,   # Marine biology
    25171353,
    37238912,
    38795522,
    45678901,
    48180672,
    65594141
]


def test_bhl_pages():
    
    print("=" * 70)
    print("TESTING NER PROCESSOR ON REAL BHL TEXTS (WITH TIMING)")
    print("=" * 70)
    print(f"Testing {len(BHL_PAGE_IDS)} BHL pages...")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    overall_start = time.time()
    
    print("\nCreating processor...")
    processor_start = time.time()
    processor = NERProcessor()
    processor_time = time.time() - processor_start
    print(f"✓ Processor created in {processor_time:.3f}s")
    
    results_summary = []
    successful = 0
    failed = 0
    
    fetch_times = []
    process_times = []
    total_chars = 0
    total_entities = 0
    
    for i, page_id in enumerate(BHL_PAGE_IDS, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(BHL_PAGE_IDS)}: PageID {page_id}")
        print(f"{'='*70}")
        
        try:
            print(f"📥 Fetching page {page_id}...")
            fetch_start = time.time()
            text = get_page_text(page_id)
            fetch_time = time.time() - fetch_start
            fetch_times.append(fetch_time)
            
            if text and len(text) > 100:
                print(f"✅ Fetched {len(text)} characters in {fetch_time:.3f}s")
                
                print(f"⚙️  Processing...")
                process_start = time.time()
                results = processor.process_text(text, method="hybrid_gn")  # Updated to use hybrid_gn
                process_time = time.time() - process_start
                process_times.append(process_time)
                
                total_chars += len(text)
                total_entities += results["total"]
                
                summary = {
                    "page_id": page_id,
                    "char_count": len(text),
                    "fetch_time": fetch_time,
                    "process_time": process_time,
                    "total_time": fetch_time + process_time,
                    "total": results["total"],
                    "spacy_count": results["spacy_count"],
                    "gliner2_count": results["gliner2_count"],
                    "gliner_count": results["gliner_count"],
                    "gnfinder_count": results["gnfinder_count"],
                    # "taxonerd_count": results["taxonerd_count"],
                    "counts": results["counts"],
                    "error": None
                }
                
                print(f"✅ Processed in {process_time:.3f}s")
                print(f"   Found {results['total']} entities")
                print(f"   spaCy: {results['spacy_count']}, GLiNER2: {results['gliner2_count']}, GLiNER: {results['gliner_count']}, GNfinder: {results['gnfinder_count']}")
                print(f"   Total time: {fetch_time + process_time:.3f}s")
                
                successful += 1
                results_summary.append(summary)
            else:
                error_msg = "No text found or text too short"
                print(f"⚠️ {error_msg}")
                failed += 1
                results_summary.append({
                    "page_id": page_id,
                    "char_count": len(text) if text else 0,
                    "fetch_time": fetch_time,
                    "process_time": 0,
                    "total_time": fetch_time,
                    "total": 0,
                    "spacy_count": 0,
                    "gliner2_count": 0,
                    "gliner_count": 0,
                    "gnfinder_count": 0,
                    # "taxonerd_count": 0,
                    "counts": {},
                    "error": error_msg
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
            results_summary.append({
                "page_id": page_id,
                "char_count": 0,
                "fetch_time": 0,
                "process_time": 0,
                "total_time": 0,
                "total": 0,
                "spacy_count": 0,
                "gliner2_count": 0,
                "gliner_count": 0,
                "gnfinder_count": 0,
                "counts": {},
                "error": str(e)
            })
    
    total_time = time.time() - overall_start
    
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'PageID':<15} {'Chars':<10} {'Fetch(s)':<10} {'Process(s)':<12} {'Total(s)':<10} {'Entities':<10}")
    print("-" * 70)
    
    for summary in results_summary:
        if not summary.get("error"):
            print(f"{summary['page_id']:<15} {summary['char_count']:<10} {summary['fetch_time']:<10.3f} {summary['process_time']:<12.3f} {summary['total_time']:<10.3f} {summary['total']:<10}")
        else:
            print(f"{summary['page_id']:<15} {summary['char_count']:<10} {summary['fetch_time']:<10.3f} {'ERROR':<12} {summary['total_time']:<10.3f} {'-':<10}")
    
    print("-" * 70)
    
    successful_results = [r for r in results_summary if not r.get("error")]
    
    print(f"\n📊 Statistics:")
    print(f"  Total pages tested: {len(BHL_PAGE_IDS)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  Total characters processed: {total_chars:,}")
    print(f"  Total entities found: {total_entities}")
    
    if successful_results:
        avg_fetch = sum(r["fetch_time"] for r in successful_results) / len(successful_results)
        avg_process = sum(r["process_time"] for r in successful_results) / len(successful_results)
        avg_total = sum(r["total_time"] for r in successful_results) / len(successful_results)
        
        print(f"\n⏱️  Average times:")
        print(f"  Fetch: {avg_fetch:.3f}s")
        print(f"  Process: {avg_process:.3f}s")
        print(f"  Total per page: {avg_total:.3f}s")
        print(f"  Overall test: {total_time:.3f}s")
        
        chars_per_second = total_chars / total_time if total_time > 0 else 0
        entities_per_second = total_entities / total_time if total_time > 0 else 0
        entities_per_page = total_entities / len(successful_results) if successful_results else 0
        
        print(f"\n📈 Performance:")
        print(f"  Characters/second: {chars_per_second:.0f}")
        print(f"  Entities/second: {entities_per_second:.1f}")
        print(f"  Average entities per page: {entities_per_page:.1f}")
        

        total_spacy = sum(r["spacy_count"] for r in successful_results)
        total_gliner2 = sum(r["gliner2_count"] for r in successful_results)
        total_gliner = sum(r["gliner_count"] for r in successful_results)
        total_gnfinder = sum(r["gnfinder_count"] for r in successful_results)  # NEW
        # total_taxonerd = sum(r["taxonerd_count"] for r in successful_results)  # NEW

        print(f"\n🤖 Model contributions:")
        print(f"  spaCy: {total_spacy} entities ({total_spacy/total_entities*100:.1f}%)")
        print(f"  GLiNER2: {total_gliner2} entities ({total_gliner2/total_entities*100:.1f}%)")
        print(f"  GLiNER: {total_gliner} entities ({total_gliner/total_entities*100:.1f}%)")
        print(f"  GNfinder: {total_gnfinder} entities ({total_gnfinder/total_entities*100:.1f}%)")  # NEW
        # print(f"  TaxoNERD: {total_taxonerd} entities ({total_taxonerd/total_entities*100:.1f}%)")  # NEW

        all_counts = {}
        for r in successful_results:
            for label, count in r["counts"].items():
                all_counts[label] = all_counts.get(label, 0) + count
        
        if all_counts:
            print(f"\n📋 Entity type distribution:")
            for label, count in sorted(all_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_entities) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ BHL Text Tests Complete!")
    print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    test_bhl_pages()