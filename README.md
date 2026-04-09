# Embedding, Clustering & Vectorization Workshop
**PROG8245 — Team Assignment**

## Team Members

| Name | Student ID |
|---|---|
| Emmanuel Ihejiamazu | 9080005 |
| Liggia Elena Taboada Cruz | 9085905 |
| Chao-Chung Liu | 9067679 |

---

## Overview
This workshop notebook introduces key techniques in modern NLP and machine learning pipelines, including text vectorization, embeddings, and clustering. It demonstrates how raw textual data can be transformed into numerical representations and analyzed using unsupervised learning methods.

The assignment notebook (`WorkshopAssignment.ipynb`) uses three Agatha Christie mystery novels as the knowledge corpus to train and compare **Word2Vec** (predictive) and **GloVe** (count-based) word embedding models.

## Learning Objectives
- Understand text vectorization methods (e.g., Bag-of-Words, TF-IDF)
- Learn how embeddings capture semantic meaning
- Apply clustering techniques to group similar data
- Explore practical workflows using Python and common ML libraries

## Contents

| File | Description |
|---|---|
| `EmbeddingClusteringVectorizationWorkshop.ipynb` | Original tutorial notebook |
| `WorkshopAssignment.ipynb` | Team assignment notebook (NLP pipeline + Word2Vec + GloVe) |
| `data/missing_will.txt` | Agatha Christie — The Missing Will |
| `data/muder_on_the_links.txt` | Agatha Christie — The Murder on the Links |
| `data/roger_ackroyd.txt` | Agatha Christie — The Murder of Roger Ackroyd |

---

## How to Run

### 1. Clone the repository
```bash
git clone <repo-url>
cd EmbeddingCusteringVectorizationWorkshop
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install jupyter nltk gensim numpy matplotlib scikit-learn python-dotenv
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

Open `WorkshopAssignment.ipynb` and run the cells from top to bottom.

> **Note:** The first run will automatically download NLTK tokenizer data into a local `nltk_data/` folder and GloVe pre-trained vectors (~66 MB) via `gensim.downloader`. Both are cached after the first download.

### 5. (Optional) OpenAI / LangChain cells
If you want to run the LangChain section in the tutorial notebook, create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

---

## Key Concepts
- **Vectorization**: Converting text into numerical features
- **Embeddings**: Dense vector representations capturing semantics
- **Clustering**: Grouping similar data points without labels

## Tools & Libraries
- Python 3.10+
- `gensim` — Word2Vec and GloVe (via downloader)
- `nltk` — tokenization and stopwords
- `scikit-learn` — dimensionality reduction (PCA/SVD), cosine similarity
- `numpy`, `matplotlib` — numerics and visualization

## Applications
- Document similarity
- Topic modeling
- Recommendation systems
- Semantic search

## Notes
This workshop is designed for learners with basic Python and machine learning knowledge.
