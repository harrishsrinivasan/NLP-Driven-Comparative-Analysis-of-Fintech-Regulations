# 📊 FinReg NLP Analysis - Streamlit App

A comprehensive Streamlit application for comparative analysis of financial technology regulations using Natural Language Processing.

## 🎯 Features

- **PDF Document Processing**: Upload up to 4 regulatory PDF documents
- **Word Frequency Analysis**: Identify most common terms and generate WordClouds
- **TF-IDF Similarity**: Compare documents using term frequency-inverse document frequency
- **BERT Semantic Similarity**: Deep semantic comparison using transformer models
- **Sentiment Analysis**: Analyze emotional tone using VADER sentiment analyzer
- **Interactive Visualizations**: Heatmaps, word clouds, and comparison charts

## 📁 Project Structure

```
finreg-nlp-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── modules/
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF extraction & text cleaning
│   ├── frequency_analyzer.py   # Word frequency & WordCloud
│   ├── similarity_analyzer.py  # TF-IDF & BERT similarity
│   ├── sentiment_analyzer.py   # VADER sentiment analysis
│   └── data_manager.py         # Session state management
└── utils/
    ├── __init__.py
    └── config.py              # Configuration constants
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

Download this project to your local machine.

### Step 2: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Windows CMD:
.\venv\Scripts\activate.bat

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Download Required Models

```powershell
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data (run Python and execute these commands)
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

**Note**: The BERT model (`bert-base-nli-mean-tokens`) will be downloaded automatically on first use.

## 🎮 Usage

### Running the Application

```powershell
# Make sure you're in the project directory
cd finreg-nlp-app

# Run the Streamlit app
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Upload Documents**
   - Click the file uploader in the sidebar
   - Select up to 4 PDF files containing regulatory documents
   - Supported format: PDF only

2. **Process Documents**
   - Click the "🚀 Process Documents" button
   - Wait for the NLP pipeline to complete (30-60 seconds for 4 documents)
   - Processing includes: text extraction, cleaning, frequency analysis, similarity computation, and sentiment analysis

3. **Explore Results**
   - Navigate through three main tabs:
     - **📈 Frequency Analysis**: View word frequencies and word clouds
     - **🔍 Similarity Analysis**: Compare documents using TF-IDF and BERT
     - **😊 Sentiment Analysis**: Analyze emotional tone of documents

4. **Clear Data**
   - Use the "🗑️ Clear All Data" button to reset and start over

## 📊 Analysis Details

### Frequency Analysis
- **Word Frequency Tables**: Top 20 most frequent words per document
- **WordClouds**: Visual representation (400×330px, max 200 words)
- **Cross-Document Comparison**: Compare word usage across documents

### Similarity Analysis
- **TF-IDF Similarity**: Statistical term-based comparison
  - Uses scikit-learn's TfidfVectorizer
  - Computes cosine similarity between document vectors
  
- **BERT Similarity**: Semantic understanding comparison
  - Uses `bert-base-nli-mean-tokens` model
  - Captures deeper semantic relationships
  
- **Visualizations**: Heatmaps with similarity scores (0-1 scale)

### Sentiment Analysis
- **VADER Sentiment**: Lexicon-based sentiment analysis
- **Classifications**: Positive, Negative, Neutral
- **Metrics**: 
  - Sentence-level sentiment distribution
  - Overall compound scores
  - Percentage breakdowns
- **Visualizations**: Pie charts and comparison bar charts

## 🛠️ Technical Details

### NLP Pipeline

1. **Text Extraction**: PyMuPDF extracts text from PDFs
2. **Tokenization**: spaCy tokenizer (handles 2-3M characters)
3. **Text Cleaning**:
   - Lowercase conversion
   - Special character removal
   - POS tagging (TextBlob)
   - Lemmatization (WordNet)
   - Stopword removal (NLTK)
4. **Analysis**: Parallel execution of frequency, similarity, and sentiment analyses

### Models Used

- **spaCy**: `en_core_web_sm` - Tokenization and NLP processing
- **BERT**: `bert-base-nli-mean-tokens` - Semantic embeddings
- **VADER**: NLTK's pretrained sentiment lexicon
- **TextBlob**: POS tagging and lemmatization
- **scikit-learn**: TF-IDF vectorization

### Performance Optimization

- **Caching**: 
  - `@st.cache_resource` for models (loaded once)
  - `@st.cache_data` for text processing functions
  - Session state for analysis results
  
- **Expected Processing Time**:
  - First run: 30-60 seconds for 4 documents (includes model loading)
  - Subsequent runs: 10-20 seconds (models cached)

## 📋 Requirements

### Core Dependencies

```
streamlit>=1.30.0
PyMuPDF>=1.23.0
spacy>=3.7.0
textblob>=0.17.0
nltk>=3.8.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
wordcloud>=1.9.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
```

## 🔧 Configuration

Edit `utils/config.py` to customize:

- `MAX_FILE_UPLOADS`: Maximum number of files (default: 4)
- `WORDCLOUD_MAX_WORDS`: Words in word cloud (default: 200)
- `TOP_N_WORDS`: Top words to display (default: 20)
- `SPACY_MAX_LENGTH`: Maximum text length (default: 3,000,000)
- Sentiment thresholds, colors, and other UI parameters

## ⚠️ Constraints & Limitations

- **No Database**: All data stored in session state (lost on page refresh)
- **No Authentication**: Direct access to application
- **No Cloud Deployment**: Designed for local execution
- **Pretrained Models Only**: No custom model training
- **File Limit**: Maximum 4 PDF documents
- **Memory**: Large documents may require significant RAM

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```powershell
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data not found**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('vader_lexicon')
   nltk.download('stopwords')
   ```

3. **Memory errors with large PDFs**
   - Process fewer documents at once
   - Increase system RAM
   - Reduce `SPACY_MAX_LENGTH` in config

4. **Slow processing**
   - First run is slower (downloads BERT model)
   - Subsequent runs use cached models
   - Ensure adequate CPU/RAM resources

5. **PDF text extraction fails**
   - Ensure PDFs contain extractable text (not just images)
   - Try OCR preprocessing if needed (not included in this app)

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [spaCy Documentation](https://spacy.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)

## 📝 License

This project is provided as-is for educational and research purposes.

## 👥 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the configuration in `utils/config.py`
3. Ensure all models are properly installed

## 🎉 Acknowledgments

Built using open-source NLP tools:
- spaCy for tokenization
- Hugging Face Transformers for BERT embeddings
- NLTK for sentiment analysis
- Streamlit for the web interface
