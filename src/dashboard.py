import sys
import os
import warnings
import logging
import io
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

st.set_page_config(
    page_title="BHL NER Dashboard",
    page_icon="🌿",
    layout="wide"
)

ENTITY_COLOURS = {
    "SPECIES": "#90EE90",
    "LOCATION": "#00FFFF",
    "DATE": "#FFD700",
    "PERSON": "#FFB6C1",
    "ORGANIZATION": "#DDA0DD",
    "HABITAT": "#D2A679",
    "ORGANISM": "#AAFF00",
}
DEFAULT_COLOUR = "#FFFF00"

EXTRACTION_METHODS = [
    "Hybrid (spaCy + GLiNER2)",
    "Hybrid (spaCy + GLiNER)",
    "Hybrid (spaCy + GLiNER + GNfinder)",
    # "Hybrid (spaCy + GLiNER + TaxoNERD)",
    "spaCy Only",
    "GLiNER2 Only",
    "GLiNER Only",
    "GNfinder Only",
    # "TaxoNERD Only"
]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.fetch_bhl import get_page_text
from src.ner_processor import NERProcessor


def highlight_entities(text, entities):
    if not entities:
        return text
    
    sorted_entities = sorted(entities, key=lambda x: len(x["text"]), reverse=True)
    replacements = []
    
    for entity in sorted_entities:
        entity_text = entity["text"]
        label = entity["label"]
        
        if len(entity_text) < 2:
            continue
        
        colour = ENTITY_COLOURS.get(label, DEFAULT_COLOUR)
        start_idx = text.lower().find(entity_text.lower())
        
        if start_idx == -1:
            continue
        
        end_idx = start_idx + len(entity_text)
        
        overlap = False
        for existing_start, existing_end, _ in replacements:
            if not (end_idx <= existing_start or start_idx >= existing_end):
                overlap = True
                break
        
        if not overlap:
            html_span = f'<span style="background-color: {colour}; padding: 2px 4px; border-radius: 3px;" title="{label}">{entity_text}</span>'
            replacements.append((start_idx, end_idx, html_span))
    
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    highlighted_text = text
    for start_idx, end_idx, html_span in replacements:
        highlighted_text = highlighted_text[:start_idx] + html_span + highlighted_text[end_idx:]
    
    return highlighted_text


def show_model_loading_status(processor, extraction_method):
    model_map = {
        "spacy": "spaCy",
        "gliner2": "GLiNER2",
        "gliner": "GLiNER",
        "gnfinder": "GNfinder",
        # "taxonerd": "TaxoNERD"
    }
    
    loaded_models = []
    for key, name in model_map.items():
        if processor.models_loaded.get(key, False):
            loaded_models.append(name)
    
    all_loaded = True
    for key, name in model_map.items():
        if name in extraction_method and not processor.models_loaded.get(key, False):
            all_loaded = False
            break
    
    return loaded_models, all_loaded


def process_and_display(text, source, method):
    processor = st.session_state.ner_processor
    
    process_start = time.time()
    results = processor.process_text(text, method=method)
    process_time = time.time() - process_start
    
    st.session_state.processed_text = text
    st.session_state.entities = results["entities"]
    st.session_state.counts = results["counts"]
    st.session_state.source = source
    st.session_state.processing_times.append(process_time)
    st.session_state.total_pages_processed += 1
    
    return results, process_time


def display_model_loading_message(processor, extraction_method):
    loaded_models, all_loaded = show_model_loading_status(processor, extraction_method)
    
    if not all_loaded and loaded_models:
        st.info(f"💡 Models ({', '.join(loaded_models)}) were loaded. Subsequent requests will be faster.")
    elif not all_loaded:
        st.info("💡 Models are loading... Subsequent requests will be faster.")


def initialise_session_state():
    defaults = {
        "processed_text": "",
        "entities": [],
        "counts": {},
        "source": "",
        "processing_times": [],
        "total_pages_processed": 0,
        "clear_counter": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


start_time = time.time()
print("=" * 60)
print("BHL NER Dashboard - Starting up...")
print("=" * 60)

initialise_session_state()

if "ner_processor" not in st.session_state:
    print("Creating NER processor...")
    create_start = time.time()
    st.session_state.ner_processor = NERProcessor()
    create_end = time.time()
    print(f"✓ NER processor created in {create_end - create_start:.2f} seconds")

    with st.spinner("Loading NER models... This may take a moment."):
        load_start = time.time()
        processor = st.session_state.ner_processor

        print("Loading spaCy model...")
        processor.load_spacy()
        print("Loading GLiNER2 model...")
        processor.load_gliner2()
        print("Loading GLiNER model...")
        processor.load_gliner()
        print("Loading GNfinder model...")
        processor.load_gnfinder()
        # print("Loading TaxoNERD model...")
        # processor.load_taxonerd()

        load_end = time.time()
        load_time = load_end - load_start

        st.session_state.processor_load_time = load_time 

        st.success(f"✓ All models loaded in {load_time:.2f} seconds")
        print(f"✓ All models loaded in {load_time:.2f} seconds")

        loaded = []
        if processor.models_loaded["spacy"]:
            loaded.append("spaCy")
        if processor.models_loaded["gliner2"]:
            loaded.append("GLiNER2")
        if processor.models_loaded["gliner"]:
            loaded.append("GLiNER")
        if processor.models_loaded["gnfinder"]:
            loaded.append("GNfinder")
        # if processor.models_loaded["taxonerd"]:
        #     loaded.append("TaxoNERD")
        print(f"Models loaded: {', '.join(loaded)}")

        # Small delay to ensure spinner message is visible
        time.sleep(0.5)

else:
    print("✓ NER processor already loaded from session state")
    # If already loaded, show quick status
    processor = st.session_state.ner_processor
    loaded = []
    if processor.models_loaded["spacy"]:
        loaded.append("spaCy")
    if processor.models_loaded["gliner2"]:
        loaded.append("GLiNER2")
    if processor.models_loaded["gliner"]:
        loaded.append("GLiNER")
    if processor.models_loaded["gnfinder"]:
        loaded.append("GNfinder")
    # if processor.models_loaded["taxonerd"]:
        # loaded.append("TaxoNERD")
    st.info(f"✅ Models ready: {', '.join(loaded)}")


# Sidebar
with st.sidebar:
    st.subheader("⚙️ Extraction Method")
    extraction_method = st.selectbox(
        "Choose the NER extraction method:",
        EXTRACTION_METHODS,
        help="Hybrid combines models for better accuracy.",
        key="extraction_method_selectbox"
    )

    st.divider()

    st.subheader("🎨 Legend")
    for label, colour in ENTITY_COLOURS.items():
        st.markdown(
            f'<span style="background-color: {colour}; padding: 2px 8px; border-radius: 3px;">{label}</span>',
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("⚡ Performance Metrics")
    
    processor_load = st.session_state.get("processor_load_time", 0)
    st.metric("Model Loading Time", f"{processor_load:.2f}s")
    
    if st.session_state.processing_times:
        last_time = st.session_state.processing_times[-1]
        st.metric("Last Processing Time", f"{last_time:.2f}s")
        
        avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times)
        st.metric("Average Processing Time", f"{avg_time:.2f}s")
        
        st.metric("Total Processed", st.session_state.total_pages_processed)

    st.divider()
    
    st.subheader("📊 Model Status")
    processor = st.session_state.ner_processor
    
    model_status = [
        ("spacy", "spaCy"),
        ("gliner2", "GLiNER2"),
        ("gliner", "GLiNER"),
        ("gnfinder", "GNfinder"),
        # ("taxonerd", "TaxoNERD")
    ]
    
    for key, name in model_status:
        if processor.models_loaded.get(key, False):
            st.success(f"✅ {name}: Loaded")
        else:
            st.info(f"⏳ {name}: Not loaded")


# Header
st.title("🌿 BHL NER Dashboard")
st.markdown("""
This tool uses **Named Entity Recognition (NER)** to extract:
- 🧬 **Species names**
- 📍 **Locations** 
- 📅 **Dates** 
- 👤 **People** 

Enter a BHL Page ID or paste text below to get started.
""")

st.divider()

# User instructions
with st.expander("📖 How to Use This Dashboard", expanded=False):
    st.markdown("""
    ### Quick Start
    1. **Fetch from BHL**: Enter a Page ID (e.g., `54850898`) and click "Fetch from BHL"
    2. **Or paste text**: Paste your own biodiversity text in the text area
    3. **Choose a method**: Select an extraction method from the sidebar
    4. **View results**: See highlighted text, entity table, and charts
    5. **Export**: Download results as CSV
    
    ### 💡 Tips
    - **Best overall**: Use "Hybrid (spaCy + GLiNER + GNfinder)" for maximum coverage
    - **Real confidence scores**: GLiNER and GNfinder provide real confidence scores
    - **First request may be slow**: Models load on first use, subsequent requests are faster
    - **Large texts**: Processing long texts may take a few seconds
    
    ### 📊 Entity Types
    - 🟢 **SPECIES**: Scientific and common species names
    - 🔵 **LOCATION**: Places, regions, countries
    - 🟡 **DATE**: Years, dates, time periods
    - 🩷 **PERSON**: People, authors, collectors
    - 🟣 **ORGANIZATION**: Institutions, societies, projects
    - 🟠 **HABITAT**: Environments, ecosystems, habitats
    - 🟢 **ORGANISM**: Plants, animals, fungi (non-species)
    """)

st.divider()


# Input section
st.subheader("📝 Input")

col1, col2 = st.columns(2)

with col1:
    st.write("**Option 1: Fetch from BHL**")
    page_id = st.text_input(
        "Enter a BHL Page ID:",
        placeholder="e.g., 54850898",
        help="Find a Page ID by opening a page on biodiversitylibrary.org",
        key=f"page_id_input_{st.session_state.get('clear_counter', 0)}"
    )
    
    if st.button("Fetch from BHL", width='stretch', key="fetch_button"):
        if not page_id.strip():
            st.warning("⚠️ Please enter a Page ID.")
        else:
            with st.spinner(f"📡 Fetching BHL page {page_id}..."):
                fetch_start = time.time()
                text = get_page_text(page_id)
                fetch_time = time.time() - fetch_start
                
                if not text:
                    st.error(f"❌ Failed to fetch text from page {page_id}.")
                else:
                    results, process_time = process_and_display(
                        text, f"BHL Page {page_id}", extraction_method
                    )
                    processor = st.session_state.ner_processor
                    
                    st.success(
                        f"✅ Fetched {len(text)} characters and found {results['total']} entities!\n\n"
                        f"⏱️ **Fetch time**: {fetch_time:.2f}s | "
                        f"**Processing time**: {process_time:.2f}s | "
                        f"**Total**: {fetch_time + process_time:.2f}s"
                    )
                    
                    display_model_loading_message(processor, extraction_method)

with col2:
    st.write("**Option 2: Paste your own text**")
    pasted_text = st.text_area(
        "Paste biodiversity text here:",
        height=150,
        placeholder="Paste any text containing species names, locations, or dates...",
        key=f"pasted_text_input_{st.session_state.get('clear_counter', 0)}"
    )
    
    if st.button("Process Text", width='stretch', key="process_button"):
        if not pasted_text.strip():
            st.warning("⚠️ Please paste some text.")
        else:
            with st.spinner(f"🧠 Processing {len(pasted_text)} characters..."):
                results, process_time = process_and_display(
                    pasted_text, "Pasted text", extraction_method
                )
                processor = st.session_state.ner_processor
                
                st.success(
                    f"✅ Found {results['total']} entities in {len(pasted_text)} characters!\n\n"
                    f"⏱️ **Processing time**: {process_time:.2f}s"
                )
                
                display_model_loading_message(processor, extraction_method)

col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 4])
with col_clear1:
    if st.button("🗑️ Clear All", width='stretch'):
        for key in ["processed_text", "entities", "counts", "source", "processing_times"]:
            if key in st.session_state:
                st.session_state[key] = [] if key in ["entities", "processing_times"] else {}
        
        st.session_state.processed_text = ""
        st.session_state.total_pages_processed = 0
        st.session_state.clear_counter += 1
        st.rerun()

st.divider()


# Results section
stored_text = st.session_state.get("processed_text", "")
entities = st.session_state.get("entities", [])
counts = st.session_state.get("counts", {})

if stored_text and entities:
    
    if st.session_state.processing_times:
        col_time1, col_time2, col_time3 = st.columns(3)
        with col_time1:
            st.metric("🕐 Processing Time", f"{st.session_state.processing_times[-1]:.2f}s")
        with col_time2:
            avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times)
            st.metric("📊 Average Time", f"{avg_time:.2f}s")
        with col_time3:
            st.metric("📄 Total Processed", st.session_state.total_pages_processed)
    
    st.divider()

    st.subheader("🎨 Highlighted Text")
    st.caption(f"Source: {st.session_state.get('source', 'Unknown')}")
    st.caption(f"Character count: {len(stored_text)}")
    
    highlighted_html = highlight_entities(stored_text, entities)
    st.markdown(highlighted_html, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("🔍 Extracted Entities")
    
    df_data = []
    for entity in entities:
        row = {
            "Entity": entity["text"],
            "Type": entity["label"],
            "Source": entity.get("source", "N/A"),
            "Start": entity.get("start", "N/A"),
            "End": entity.get("end", "N/A"),
        }
        if "score" in entity:
            row["Score"] = f"{entity['score']:.4f}" if isinstance(entity['score'], (int, float)) else entity['score']
        df_data.append(row)

    df = pd.DataFrame(df_data).drop_duplicates(subset=["Entity", "Type"])
    st.dataframe(df, width='stretch', hide_index=True)
    
    st.write("**Summary:**")
    tags_html = ""
    for label, count in counts.items():
        colour = ENTITY_COLOURS.get(label, DEFAULT_COLOUR)
        tags_html += f'<span style="background-color: {colour}; padding: 4px 12px; border-radius: 12px; font-weight: bold; display: inline-block; margin: 4px 4px;">{label}: {count}</span> '
    st.markdown(tags_html, unsafe_allow_html=True)
    
    st.subheader("📊 Entity Type Distribution")
    
    if counts:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.keys(), counts.values(), color='#00897B')
        ax.set_xlabel('Entity Type')
        ax.set_ylabel('Count')
        ax.set_title('Entities Found')
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        st.pyplot(fig)
    else:
        st.info("No data to display.")

    st.write("**Download Results:**")
    df_export = pd.DataFrame(df_data)
    csv_buffer = io.StringIO()
    df_export.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_data,
        file_name="bhl_ner_entities.csv",
        mime="text/csv",
        width='stretch'
    )

else:
    st.info("💡 No text processed yet. Paste text or fetch a BHL page using the buttons above.")


st.divider()
st.caption("🌿 BHL NER Dashboard | Built with Streamlit")