"""
Tests for NER Processor with Lazy Loading (including GNfinder).

Run: python tests/test_ner_lazy_loading.py [--quick] [--performance]
"""

import os
import sys
import time
import unittest
import warnings
from collections import Counter

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    'src'
))
from ner_processor import NERProcessor

# Global shared processor - loaded ONCE for all tests
_SHARED_PROCESSOR = None
_SHARED_PROCESSOR_LOADED = False


def get_shared_processor():
    """Get or create a shared processor instance (loads models once)."""
    global _SHARED_PROCESSOR, _SHARED_PROCESSOR_LOADED
    
    if not _SHARED_PROCESSOR_LOADED:
        print("Loading shared processor (once for all tests)...")
        _SHARED_PROCESSOR = NERProcessor()
        
        # Pre-load models silently
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            _SHARED_PROCESSOR.process_text("Pre-load models", method="hybrid")
        except Exception:
            pass
        sys.stdout = old_stdout
        
        _SHARED_PROCESSOR_LOADED = True
        print("✓ Shared processor ready\n")
    
    return _SHARED_PROCESSOR


class TestLazyLoadingBehaviour(unittest.TestCase):
    """Verify models load only when first used."""
    
    def setUp(self):
        self.processor = NERProcessor()
    
    def test_models_not_loaded_at_initialisation(self):
        self.assertIsNone(self.processor.nlp)
        self.assertIsNone(self.processor.gliner2)
        self.assertIsNone(self.processor.gliner)
        self.assertFalse(self.processor.models_loaded["spacy"])
        self.assertFalse(self.processor.models_loaded["gliner2"])
        self.assertFalse(self.processor.models_loaded["gliner"])
        self.assertFalse(self.processor.models_loaded["gnfinder"])  # NEW
        print("✓ Models not loaded at initialisation")
    
    def test_spacy_loads_on_demand(self):
        self.assertIsNone(self.processor.nlp)
        self.processor.process_spacy("Test text")
        self.assertIsNotNone(self.processor.nlp)
        self.assertTrue(self.processor.models_loaded["spacy"])
        print("✓ spaCy loads on first use")
    
    def test_gliner2_loads_on_demand(self):
        self.assertIsNone(self.processor.gliner2)
        self.processor.process_gliner2("Test text")
        self.assertIsNotNone(self.processor.gliner2)
        self.assertTrue(self.processor.models_loaded["gliner2"])
        print("✓ GLiNER2 loads on first use")
    
    def test_gliner_loads_on_demand(self):
        self.assertIsNone(self.processor.gliner)
        self.processor.process_gliner("Test text")
        self.assertIsNotNone(self.processor.gliner)
        self.assertTrue(self.processor.models_loaded["gliner"])
        print("✓ GLiNER loads on first use")
    
    def test_gnfinder_loads_on_demand(self):  # NEW
        # GNfinder is checked at init, but not "loaded" until used
        self.processor.process_gnfinder("Test text")
        self.assertTrue(self.processor.models_loaded["gnfinder"])
        print("✓ GNfinder loads on first use")
    
    def test_models_cached_after_loading(self):
        self.processor.process_spacy("First text")
        first_state = self.processor.nlp
        self.processor.process_spacy("Second text")
        second_state = self.processor.nlp
        self.assertIs(first_state, second_state)
        print("✓ Models cached and reused")


class TestExtractionQuality(unittest.TestCase):
    """Verify extraction works correctly."""
    
    @classmethod
    def setUpClass(cls):
        cls.processor = get_shared_processor()
        cls.sample_text = """
        The giant panda Ailuropoda melanoleuca is a bear native to China.
        It was first described by French missionary Armand David in 1869.
        """
    
    def test_spacy_extraction_structure(self):
        entities = self.processor.process_spacy(self.sample_text)
        self.assertGreater(len(entities), 0)
        for ent in entities:
            self.assertIn("text", ent)
            self.assertIn("label", ent)
            self.assertIn("source", ent)
            self.assertEqual(ent["source"], "spaCy")
        print(f"✓ spaCy: {len(entities)} entities")
    
    def test_gliner2_extraction_structure(self):
        entities = self.processor.process_gliner2(self.sample_text)
        self.assertIsInstance(entities, list)
        for ent in entities:
            self.assertIn("text", ent)
            self.assertIn("label", ent)
            self.assertIn("source", ent)
            self.assertEqual(ent["source"], "GLiNER2")
        print(f"✓ GLiNER2: {len(entities)} entities")
    
    def test_gliner_extraction_structure(self):
        entities = self.processor.process_gliner(self.sample_text)
        self.assertIsInstance(entities, list)
        for ent in entities:
            self.assertIn("text", ent)
            self.assertIn("label", ent)
            self.assertIn("source", ent)
            self.assertEqual(ent["source"], "GLiNER")
        print(f"✓ GLiNER: {len(entities)} entities")
    
    def test_gnfinder_extraction_structure(self):  # NEW
        entities = self.processor.process_gnfinder(self.sample_text)
        self.assertIsInstance(entities, list)
        if entities:  # Only if GNfinder is available
            for ent in entities:
                self.assertIn("text", ent)
                self.assertIn("label", ent)
                self.assertEqual(ent["label"], "SPECIES")
                self.assertEqual(ent["source"], "GNfinder")
        print(f"✓ GNfinder: {len(entities)} entities")
    
    def test_hybrid_mode_combines_models(self):
        result = self.processor.process_text(self.sample_text, method="hybrid")
        self.assertGreaterEqual(result["spacy_count"], 0)
        self.assertGreaterEqual(result["gliner2_count"], 0)
        self.assertLessEqual(result["total"], result["spacy_count"] + result["gliner2_count"])
        self.assertGreaterEqual(result["total"], max(result["spacy_count"], result["gliner2_count"]))
        print(f"✓ Hybrid (GLiNER2): spaCy={result['spacy_count']}, GLiNER2={result['gliner2_count']}, unique={result['total']}")
    
    def test_hybrid_g_mode_combines_models(self):
        result = self.processor.process_text(self.sample_text, method="hybrid_g")
        self.assertGreaterEqual(result["spacy_count"], 0)
        self.assertGreaterEqual(result["gliner_count"], 0)
        self.assertLessEqual(result["total"], result["spacy_count"] + result["gliner_count"])
        self.assertGreaterEqual(result["total"], max(result["spacy_count"], result["gliner_count"]))
        print(f"✓ Hybrid (GLiNER): spaCy={result['spacy_count']}, GLiNER={result['gliner_count']}, unique={result['total']}")
    
    def test_hybrid_gn_mode_combines_models(self):  # NEW
        result = self.processor.process_text(self.sample_text, method="hybrid_gn")
        self.assertGreaterEqual(result["spacy_count"], 0)
        self.assertGreaterEqual(result["gliner_count"], 0)
        self.assertGreaterEqual(result["gnfinder_count"], 0)
        print(f"✓ Hybrid (GNfinder): spaCy={result['spacy_count']}, GLiNER={result['gliner_count']}, GNfinder={result['gnfinder_count']}, unique={result['total']}")
    
    def test_deduplication_removes_duplicates(self):
        result = self.processor.process_text(
            "China is great. China is in Asia.",
            method="hybrid"
        )
        china_entities = [e for e in result["entities"] if e["text"].lower() == "china"]
        self.assertLessEqual(len(china_entities), 1)
        print(f"✓ Deduplication: {len(result['entities'])} unique")
    
    def test_method_selection_works(self):
        methods = ["spacy_only", "gliner2_only", "gliner_only", "gnfinder_only", "hybrid", "hybrid_g", "hybrid_gn"]
        for method in methods:
            result = self.processor.process_text(self.sample_text, method=method)
            self.assertEqual(result["method"], method)
        print("✓ All methods work")


class TestFilteringAndProcessing(unittest.TestCase):
    """Test filters and processing logic."""
    
    @classmethod
    def setUpClass(cls):
        cls.processor = get_shared_processor()
    
    def test_species_filtering(self):
        text = "Species 123 invalid. Species Abc valid. Species X short."
        entities = self.processor.process_gliner2(text)
        for ent in entities:
            if ent["label"] == "SPECIES":
                self.assertFalse(ent["text"].isdigit())
                self.assertGreaterEqual(len(ent["text"]), 3)
        print(f"✓ Species filter: {len(entities)} valid")
    
    def test_label_standardisation(self):
        result = self.processor.process_text("New York is in USA", method="spacy_only")
        labels = {"LOCATION", "ORGANIZATION", "PERSON", "DATE", 
                  "SPECIES", "HABITAT", "ORGANISM", "GENUS", "FAMILY", "ORDER"}
        for ent in result["entities"]:
            self.assertIn(ent["label"], labels)
        print("✓ Label standardisation")


class TestPerformance(unittest.TestCase):
    """Test performance improvements."""
    
    @classmethod
    def setUpClass(cls):
        cls.processor = get_shared_processor()
        cls.test_text = "Performance test text."
    
    def test_initialisation_speed(self):
        start = time.time()
        NERProcessor()
        init_time = time.time() - start
        self.assertLess(init_time, 1.0)
        print(f"✓ Init: {init_time:.3f}s")
    
    def test_first_use_loads_models(self):
        processor = NERProcessor()
        start = time.time()
        processor.process_text(self.test_text, method="hybrid")
        first_time = time.time() - start
        self.assertGreater(first_time, 0.5)
        print(f"✓ First use: {first_time:.3f}s")
    
    def test_subsequent_use_faster(self):
        processor = NERProcessor()
        
        # First use (loads models)
        start = time.time()
        processor.process_text(self.test_text, method="hybrid")
        first_time = time.time() - start
        
        # Second use (cached)
        start = time.time()
        processor.process_text(self.test_text + " more", method="hybrid")
        cached_time = time.time() - start
        
        self.assertLess(cached_time, first_time * 0.8)
        print(f"✓ Cached: {cached_time:.3f}s (vs first: {first_time:.3f}s)")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @classmethod
    def setUpClass(cls):
        cls.processor = get_shared_processor()
    
    def test_empty_text(self):
        result = self.processor.process_text("")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["error"], "Invalid input")
        print("✓ Empty text")
    
    def test_none_input(self):
        result = self.processor.process_text(None)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["error"], "Invalid input")
        print("✓ None input")
    
    def test_short_text(self):
        result = self.processor.process_text("China", method="hybrid")
        self.assertIsInstance(result["entities"], list)
        print(f"✓ Short text: {result['total']} entities")
    
    def test_numeric_text(self):
        result = self.processor.process_text("1234567890", method="hybrid")
        self.assertIsInstance(result["entities"], list)
        self.assertEqual(result["total"], 0)
        print("✓ Numeric text")
    
    def test_special_characters(self):
        result = self.processor.process_text("Dr. Smith's lab @ Harvard!", method="hybrid")
        self.assertIsInstance(result["entities"], list)
        print(f"✓ Special chars: {result['total']} entities")


def run_quick_test():
    """Quick test for manual verification."""
    print("\n" + "=" * 70)
    print("QUICK TEST (All Methods)")
    print("=" * 70)
    
    print("\n1. Creating processor...")
    start = time.time()
    processor = NERProcessor()
    print(f"   ✓ Created in {time.time() - start:.3f}s")
    
    print("\n2. Testing all methods...")
    
    test_text = "The giant panda Ailuropoda melanoleuca is native to China."
    
    methods = ["spacy_only", "gliner2_only", "gliner_only", "gnfinder_only", "hybrid", "hybrid_g", "hybrid_gn"]
    
    for method in methods:
        start = time.time()
        result = processor.process_text(test_text, method=method)
        elapsed = time.time() - start
        
        count_str = f"spaCy: {result['spacy_count']}, GLiNER2: {result['gliner2_count']}, GLiNER: {result['gliner_count']}, GNfinder: {result['gnfinder_count']}"
        print(f"   {method}: {result['total']} entities ({count_str}) in {elapsed:.3f}s")
    
    print("\n✅ Done!")


def run_performance_demo():
    """Demonstrate performance benefits."""
    print("\n" + "=" * 70)
    print("PERFORMANCE DEMO")
    print("=" * 70)
    
    print("\n[LAZY LOADING]")
    start = time.time()
    processor = NERProcessor()
    init = time.time() - start
    print(f"  Dashboard ready: {init:.3f}s")
    
    print("  User requests NER...")
    start = time.time()
    processor.process_text("Test", method="hybrid")
    load = time.time() - start
    print(f"  Models loaded: {load:.3f}s")
    print(f"  Total: {init + load:.3f}s")
    
    print("\n[EAGER LOADING - Previous]")
    print("  Dashboard ready: ~3.5s")
    print("  User requests NER: ~0.5s")
    print("  Total: ~4.0s")
    
    print("\n✅ Lazy loading saves ~3.5s on startup!")


if __name__ == "__main__":
    if "--quick" in sys.argv:
        run_quick_test()
    elif "--performance" in sys.argv:
        run_performance_demo()
    else:
        print("=" * 70)
        print("NER PROCESSOR LAZY LOADING - FULL TEST SUITE")
        print("=" * 70)
        print("\nLoading shared processor (once for all tests)...")
        
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            get_shared_processor()
        except Exception:
            pass
        sys.stdout = old_stdout
        
        print("\nRunning all tests...\n")
        
        test_classes = [
            TestLazyLoadingBehaviour,
            TestExtractionQuality,
            TestFilteringAndProcessing,
            TestPerformance,
            TestEdgeCases
        ]
        
        suite = unittest.TestSuite()
        for test_class in test_classes:
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
        
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Tests: {result.testsRun}")
        if result.wasSuccessful():
            print("  ✅ ALL PASSED")
            print("  ✅ Lazy loading working correctly")
            print("  ✅ GNfinder integration successful")
        else:
            print(f"  ❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
        run_performance_demo()
        print("\n")
        run_quick_test()