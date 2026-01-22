"""
Configuration file for FinReg NLP Analysis Streamlit App
Contains constants, model names, and UI parameters
"""

# Model Configuration
SPACY_MODEL = "en_core_web_sm"
BERT_MODEL = "bert-base-nli-mean-tokens"

# spaCy Configuration
SPACY_MAX_LENGTH = 3_000_000  # Maximum text length for spaCy processing

# WordCloud Configuration
WORDCLOUD_WIDTH = 400
WORDCLOUD_HEIGHT = 330
WORDCLOUD_MAX_WORDS = 200
WORDCLOUD_BACKGROUND = "white"

# Display Configuration
TOP_N_WORDS = 20  # Number of top frequent words to display
FIGURE_SIZE = (10, 8)  # Default figure size for plots

# File Upload Configuration
MAX_FILE_UPLOADS = 4
ALLOWED_FILE_TYPES = ["pdf"]
MAX_FILE_SIZE_MB = 50

# Sentiment Thresholds (VADER)
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# UI Text
APP_TITLE = "📊 FinReg NLP Analysis"
APP_SUBTITLE = "Comparative Analysis of Financial Technology Regulations"
SIDEBAR_HEADER = "📁 Document Upload"

# Color Schemes
HEATMAP_COLORMAP = "YlGnBu"
SENTIMENT_COLORS = ["#90EE90", "#FFB6C1", "#D3D3D3"]  # Positive, Negative, Neutral

# Processing Messages
MSG_UPLOADING = "Processing uploaded documents..."
MSG_EXTRACTING = "Extracting text from PDFs..."
MSG_CLEANING = "Cleaning and preprocessing text..."
MSG_ANALYZING_FREQ = "Analyzing word frequencies..."
MSG_COMPUTING_TFIDF = "Computing TF-IDF similarities..."
MSG_COMPUTING_BERT = "Computing BERT embeddings (this may take a while)..."
MSG_ANALYZING_SENTIMENT = "Analyzing sentiment..."
MSG_COMPLETE = "✅ Analysis complete!"
