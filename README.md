# 🌿 BHL NER Dashboard

Named Entity Recognition for Biodiversity Heritage Library texts.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

---

## 📌 Overview

This dashboard applies **Named Entity Recognition (NER)** to texts from the **Biodiversity Heritage Library (BHL)**. It extracts species names, locations, dates, people, and more from historical biodiversity literature.

**Key Entities Extracted:**
- 🧬 Species names (scientific and common)
- 📍 Locations (countries, regions, habitats)
- 📅 Dates (years, historical periods)
- 👤 People (authors, collectors, naturalists)
- 🏢 Organisations (institutions, societies)
- 🌿 Habitats (ecosystems, environments)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **4 NER Models** | spaCy, GLiNER2, GLiNER, GNfinder |
| **7 Extraction Methods** | Single models + hybrid combinations |
| **BHL Integration** | Fetch texts by Page ID |
| **Custom Input** | Paste your own text |
| **Entity Highlighting** | Colour-coded entities |
| **Visualisation** | Bar chart of entity types |
| **CSV Export** | Download results |
| **Performance Metrics** | Loading and processing times |

---

## 🧠 Extraction Methods

| Method | Models | Best For |
|--------|--------|----------|
| Hybrid (spaCy + GLiNER2) | spaCy + GLiNER2 | General extraction |
| Hybrid (spaCy + GLiNER) | spaCy + GLiNER | General + domain entities |
| **Hybrid (spaCy + GLiNER + GNfinder)** | spaCy + GLiNER + GNfinder | **Maximum coverage** |
| spaCy Only | spaCy | Speed |
| GLiNER2 Only | GLiNER2 | Zero-shot entities |
| GLiNER Only | GLiNER | Zero-shot + confidence scores |
| GNfinder Only | GNfinder | Species names only |

> **Default Method**: The dashboard uses **"Hybrid (spaCy + GLiNER2)"** by default. This provides a good balance of speed and accuracy for general use.

### How Hybrid Works

1. **Independent Processing**: Each model processes the text independently
2. **Collection**: All entities are collected into a single list
3. **Overlap Merging**: Overlapping entities are merged, keeping the best one:
   - Domain-specific models (GLiNER2, GLiNER, GNfinder) take priority over spaCy
   - More specific labels (SPECIES > LOCATION > PERSON) take priority
   - Longer text (more complete entity) is preferred
4. **Deduplication**: Entities are deduplicated using:
   - Key = `lowercase_text + "|" + label`
   - Example: `"ailuropoda melanoleuca|SPECIES"`
5. The `source` attribute tracks which model found each entity
6. This maximises coverage while avoiding duplicates and partial matches

---

## 📦 Setup

### Prerequisites
- Python 3.13+

### Installation

Clone the repository:

    git clone https://github.com/[your-username]/bhl-ner-dashboard.git
    cd bhl-ner-dashboard

Install dependencies:

    py -m pip install -r requirements.txt

Download spaCy model:

    py -m spacy download en_core_web_sm

Install GNfinder (optional but recommended):

    Download from: https://github.com/gnames/gnfinder/releases
    Place gnfinder.exe in C:\bin or add to PATH

Run the dashboard:

    py -m streamlit run src/dashboard.py

Open http://localhost:8501 in your browser.

---

## 📖 Usage

### Fetch from BHL

1. Enter a Page ID (e.g., `54850898`)
2. Click **"Fetch from BHL"**
3. View highlighted text and entities

### Paste Text

1. Paste your own biodiversity text
2. Click **"Process Text"**
3. View results

### Results

- **Highlighted Text**: Colour-coded entities
- **Entity Table**: List of all entities
- **Bar Chart**: Entity type distribution
- **CSV Export**: Download results

---

## 📋 Dependencies

    streamlit>=1.28.0
    pandas>=2.0.0
    numpy>=1.24.0
    matplotlib>=3.7.0
    plotly>=5.17.0
    requests>=2.31.0
    spacy>=3.7.0
    gliner>=0.2.0
    gliner2>=1.0.0
    torch>=2.0.0
    torchvision>=0.15.0
    transformers>=4.30.0
    python-dotenv>=1.0.0

---

## 🧪 Testing

Run the test suite to verify everything is working:

    py tests/test_extraction_methods.py
    py tests/test_bhl_texts.py
    py tests/test_ner_lazy_loading.py --quick

All tests are passing (25/25).

---

## 🔌 Plugin System

The dashboard uses a **plugin system** that automatically discovers NER models. To add a new model, you don't need to edit any existing code.

### How to Add a New Model

1. Create a new Python file in `src/models/` (e.g., `my_model.py`)

2. Inherit from `BaseNERModel` and implement four things:

```python
from .base import BaseNERModel

class MyModel(BaseNERModel):
    
    @property
    def name(self):
        return "My Model"          # Display name
    
    @property
    def key(self):
        return "my_model"          # Unique identifier
    
    def load(self):
        # Load your model here
        self.loaded = True
    
    def extract(self, text):
        # Extract entities from text
        # Return list of dicts with: text, label, source, start, end, score (optional)
        entities = []
        return entities
```

3. Restart the dashboard - your model is automatically discovered!

### How It Works

The system scans the `src/models/` folder on startup and loads any class that inherits from `BaseNERModel`. No changes to `ner_processor.py` or `dashboard.py` are required.

---

## 👏 Acknowledgements

- **[Xia Cui](https://www.mmu.ac.uk/staff/profile/dr-xia-cui)** - Internship Supervisor, Senior Lecturer in Software Engineering, Manchester Metropolitan University
- [Biodiversity Heritage Library](https://www.biodiversitylibrary.org/)
- [spaCy](https://spacy.io/)
- [GLiNER](https://github.com/urchade/GLiNER)
- [GNfinder](https://github.com/gnames/gnfinder)
