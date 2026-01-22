"""
Sentiment Analysis Module
Handles VADER sentiment analysis and visualization
"""

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from utils.config import POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD, SENTIMENT_COLORS


@st.cache_resource
def load_vader_analyzer():
    """
    Initialize and cache VADER sentiment analyzer.
    Downloads required NLTK data if not present.
    
    Returns:
        SentimentIntensityAnalyzer: Initialized VADER analyzer
    """
    try:
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        # Initialize analyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer
    
    except Exception as e:
        st.error(f"❌ Error loading VADER analyzer: {str(e)}")
        st.stop()


@st.cache_data(show_spinner=False)
def analyze_document_sentiment(text, doc_name):
    """
    Analyze sentiment of a document using VADER.
    
    Args:
        text: Document text (raw, not cleaned)
        doc_name: Name of the document
    
    Returns:
        dict: Sentiment analysis results
    """
    try:
        analyzer = load_vader_analyzer()
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return {
                'doc_name': doc_name,
                'total_sentences': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'positive_pct': 0.0,
                'negative_pct': 0.0,
                'neutral_pct': 0.0,
                'avg_compound': 0.0
            }
        
        # Analyze each sentence
        pos_count = 0
        neg_count = 0
        neu_count = 0
        compound_scores = []
        
        for sentence in sentences:
            scores = analyzer.polarity_scores(sentence)
            compound = scores['compound']
            compound_scores.append(compound)
            
            if compound >= POSITIVE_THRESHOLD:
                pos_count += 1
            elif compound <= NEGATIVE_THRESHOLD:
                neg_count += 1
            else:
                neu_count += 1
        
        total = len(sentences)
        
        return {
            'doc_name': doc_name,
            'total_sentences': total,
            'positive': pos_count,
            'negative': neg_count,
            'neutral': neu_count,
            'positive_pct': (pos_count / total) * 100 if total > 0 else 0,
            'negative_pct': (neg_count / total) * 100 if total > 0 else 0,
            'neutral_pct': (neu_count / total) * 100 if total > 0 else 0,
            'avg_compound': sum(compound_scores) / len(compound_scores) if compound_scores else 0
        }
    
    except Exception as e:
        st.error(f"❌ Error analyzing sentiment for {doc_name}: {str(e)}")
        return None


def create_sentiment_pie_chart(sentiment_result):
    """
    Create pie chart for sentiment distribution.
    
    Args:
        sentiment_result: Sentiment analysis result dictionary
    
    Returns:
        matplotlib.figure.Figure: Pie chart figure
    """
    try:
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [
            sentiment_result['positive_pct'],
            sentiment_result['negative_pct'],
            sentiment_result['neutral_pct']
        ]
        
        # Filter out zero values
        non_zero = [(label, size, color) for label, size, color in zip(labels, sizes, SENTIMENT_COLORS) if size > 0]
        
        if not non_zero:
            return None
        
        labels, sizes, colors = zip(*non_zero)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax.set_title(
            f"Sentiment Distribution - {sentiment_result['doc_name']}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating pie chart: {str(e)}")
        return None


def create_sentiment_bar_chart(sentiment_results_list):
    """
    Create comparison bar chart for multiple documents.
    
    Args:
        sentiment_results_list: List of sentiment result dictionaries
    
    Returns:
        matplotlib.figure.Figure: Bar chart figure
    """
    try:
        doc_names = [r['doc_name'] for r in sentiment_results_list]
        positive = [r['positive_pct'] for r in sentiment_results_list]
        negative = [r['negative_pct'] for r in sentiment_results_list]
        neutral = [r['neutral_pct'] for r in sentiment_results_list]
        
        x = range(len(doc_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar([i - width for i in x], positive, width, label='Positive', color=SENTIMENT_COLORS[0])
        ax.bar(x, negative, width, label='Negative', color=SENTIMENT_COLORS[1])
        ax.bar([i + width for i in x], neutral, width, label='Neutral', color=SENTIMENT_COLORS[2])
        
        ax.set_xlabel('Documents', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Sentiment Distribution Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(doc_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating bar chart: {str(e)}")
        return None


def display_sentiment_analysis(sentiment_result):
    """
    Display sentiment analysis results for a single document.
    
    Args:
        sentiment_result: Sentiment analysis result dictionary
    """
    st.subheader(f"😊 {sentiment_result['doc_name']}")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sentences", f"{sentiment_result['total_sentences']:,}")
    with col2:
        st.metric("Positive", f"{sentiment_result['positive_pct']:.1f}%", 
                 delta=f"{sentiment_result['positive']} sentences")
    with col3:
        st.metric("Negative", f"{sentiment_result['negative_pct']:.1f}%",
                 delta=f"{sentiment_result['negative']} sentences")
    with col4:
        st.metric("Neutral", f"{sentiment_result['neutral_pct']:.1f}%",
                 delta=f"{sentiment_result['neutral']} sentences")
    
    # Display sentiment interpretation
    avg_compound = sentiment_result['avg_compound']
    if avg_compound >= POSITIVE_THRESHOLD:
        sentiment_label = "✅ Overall Positive"
        sentiment_color = "green"
    elif avg_compound <= NEGATIVE_THRESHOLD:
        sentiment_label = "❌ Overall Negative"
        sentiment_color = "red"
    else:
        sentiment_label = "⚖️ Overall Neutral"
        sentiment_color = "gray"
    
    st.markdown(f"**Overall Sentiment:** :{sentiment_color}[{sentiment_label}] (Compound Score: {avg_compound:.3f})")
    
    # Display pie chart
    fig = create_sentiment_pie_chart(sentiment_result)
    if fig:
        st.pyplot(fig)
        plt.close()


def display_sentiment_comparison(sentiment_results_list):
    """
    Display comparative sentiment analysis for multiple documents.
    
    Args:
        sentiment_results_list: List of sentiment result dictionaries
    """
    st.subheader("📊 Sentiment Comparison")
    
    # Create summary table
    summary_data = []
    for result in sentiment_results_list:
        summary_data.append({
            'Document': result['doc_name'],
            'Positive %': f"{result['positive_pct']:.1f}%",
            'Negative %': f"{result['negative_pct']:.1f}%",
            'Neutral %': f"{result['neutral_pct']:.1f}%",
            'Avg Compound': f"{result['avg_compound']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Display comparison bar chart
    fig = create_sentiment_bar_chart(sentiment_results_list)
    if fig:
        st.pyplot(fig)
        plt.close()


def export_sentiment_data(sentiment_results_list, filename="sentiment_analysis.csv"):
    """
    Export sentiment results to CSV format.
    
    Args:
        sentiment_results_list: List of sentiment result dictionaries
        filename: Output filename
    
    Returns:
        str: CSV data as string
    """
    df = pd.DataFrame(sentiment_results_list)
    return df.to_csv(index=False)


def get_sentiment_summary(sentiment_results_list):
    """
    Get a text summary of sentiment analysis across all documents.
    
    Args:
        sentiment_results_list: List of sentiment result dictionaries
    
    Returns:
        str: Summary text
    """
    if not sentiment_results_list:
        return "No sentiment data available."
    
    avg_positive = sum(r['positive_pct'] for r in sentiment_results_list) / len(sentiment_results_list)
    avg_negative = sum(r['negative_pct'] for r in sentiment_results_list) / len(sentiment_results_list)
    avg_neutral = sum(r['neutral_pct'] for r in sentiment_results_list) / len(sentiment_results_list)
    
    summary = f"""
    **Overall Sentiment Summary:**
    - Average Positive: {avg_positive:.1f}%
    - Average Negative: {avg_negative:.1f}%
    - Average Neutral: {avg_neutral:.1f}%
    
    Most positive document: {max(sentiment_results_list, key=lambda x: x['positive_pct'])['doc_name']}
    Most negative document: {max(sentiment_results_list, key=lambda x: x['negative_pct'])['doc_name']}
    """
    
    return summary
