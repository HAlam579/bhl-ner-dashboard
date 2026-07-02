"""
Test script for Week 6 features: Bar Chart and CSV Export.
Run with: py tests/test_week6.py
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import io

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("Testing Week 6 Features")
print("=" * 60)

# ============================================================================
# TEST 1: Create a sample DataFrame (simulating entities)
# ============================================================================

print("\n--- Test 1: Sample Data ---")

sample_entities = [
    {"Entity": "Ailuropoda melanoleuca", "Type": "SPECIES", "Source": "GLiNER2"},
    {"Entity": "China", "Type": "LOCATION", "Source": "GLiNER2"},
    {"Entity": "giant panda", "Type": "SPECIES", "Source": "GLiNER2"},
    {"Entity": "Ailuropoda", "Type": "PERSON", "Source": "spaCy"},
    {"Entity": "China", "Type": "GPE", "Source": "spaCy"},
]

sample_counts = {
    "SPECIES": 2,
    "LOCATION": 1,
    "PERSON": 1,
    "GPE": 1,
}

df = pd.DataFrame(sample_entities).drop_duplicates(subset=["Entity", "Type"])
print(f"✓ Sample DataFrame created with {len(df)} rows")
print(df.to_string(index=False))

# ============================================================================
# TEST 2: Bar Chart (Visual Test)
# ============================================================================

print("\n--- Test 2: Bar Chart ---")

if sample_counts:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sample_counts.keys(), sample_counts.values(), color='#2E7D32')
    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Count')
    ax.set_title('Entities Found')
    ax.tick_params(axis='x', rotation=45)
    
    # Display the chart
    plt.tight_layout()
    plt.savefig('test_chart.png')
    print("✓ Bar chart created and saved as 'test_chart.png'")
    print("  Please open 'test_chart.png' to verify it looks correct.")
else:
    print("✗ No data to display.")

# ============================================================================
# TEST 3: CSV Export
# ============================================================================

print("\n--- Test 3: CSV Export ---")

# Create CSV
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

print(f"✓ CSV created with {len(csv_data)} characters")
print("\nCSV Preview:")
print("-" * 40)
print(csv_data[:500])
print("-" * 40)

# Save to file for inspection
with open('test_entities.csv', 'w', encoding='utf-8') as f:
    f.write(csv_data)
print("\n✓ CSV saved as 'test_entities.csv'")
print("  Please open it in Excel or a text editor to verify the format.")

# ============================================================================
# TEST 4: Verify the Data
# ============================================================================

print("\n--- Test 4: Data Verification ---")

# Check that species are correctly labelled
species_count = df[df["Type"] == "SPECIES"].shape[0]
expected_species = 2
print(f"Species found: {species_count} (Expected: {expected_species})")
if species_count == expected_species:
    print("  ✅ Species count matches expected")
else:
    print("  ❌ Species count does not match expected")

# Check that locations are correctly labelled
location_count = df[df["Type"].isin(["LOCATION", "GPE"])].shape[0]
expected_locations = 2
print(f"Locations found: {location_count} (Expected: {expected_locations})")
if location_count == expected_locations:
    print("  ✅ Location count matches expected")
else:
    print("  ❌ Location count does not match expected")

print("\n" + "=" * 60)
print("Test complete. Check 'test_chart.png' and 'test_entities.csv'.")
print("=" * 60)