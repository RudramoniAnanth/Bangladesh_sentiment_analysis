import streamlit as st
import nltk
import re
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('Punkt', force=True)  # Add force=True here

nltk.download('Stopwords', force=True)

# Streamlit app settings
st.set_page_config(page_title="Wikipedia Sentiment Analyzer", layout="wide")
st.title("📘 Wikipedia Text Sentiment Analyzer")
st.markdown("""
Enter a paragraph of text (like from a Wikipedia article) and this app will:
- Break it into sentences
- Analyze sentiment (positive, negative, neutral)
- Show you graphs, word cloud, and summary stats
""")

# Text input
user_input = st.text_area("✍️ Enter text to analyze:", height=300)

if user_input:
    # --- TEXT PREPROCESSING ---
    text = re.sub(r'\[.*?\]', '', user_input)  # Remove references like [1]
    text = re.sub(r'\s+', ' ', text).strip()

    # --- SENTENCE TOKENIZATION + SENTIMENT ---
    sentences = sent_tokenize(text)
    df = pd.DataFrame(sentences, columns=['Sentence'])
    df['Polarity'] = df['Sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Sentiment'] = df['Polarity'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

    st.subheader("📊 Sentiment Analysis Table")
    st.dataframe(df)

    # --- WORD PROCESSING ---
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stopwords.words('english')]
    word_freq = Counter(words)
    common_words = word_freq.most_common(20)

    # --- WORD COUNT TABLE + BAR CHART (2 COLUMNS) ---
    st.divider()
    st.subheader("📌 Top Words Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔤 Top 20 Words Table")
        top_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
        st.dataframe(top_words_df)

    with col2:
        st.markdown("#### 📊 Top 20 Frequent Words (Bar Chart)")
        fig_barh, ax_barh = plt.subplots(figsize=(8, 6))
        ax_barh.barh(top_words_df['Word'], top_words_df['Frequency'], color='skyblue')
        ax_barh.set_xlabel('Frequency')
        ax_barh.set_ylabel('Words')
        ax_barh.set_title('Top 20 Frequent Words')
        ax_barh.invert_yaxis()
        st.pyplot(fig_barh)
   
    # Sentiment Distribution Plot
    st.subheader("📈 Sentiment Distribution")
    col1, col2, col3 = st.columns([1, 2, 1])  # Center the plot
    with col2:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sentiment_counts = df['Sentiment'].value_counts()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="pastel", ax=ax1)
        ax1.set_ylabel("Number of Sentences")
        ax1.set_title("Sentiment Breakdown")
        st.pyplot(fig1)
        
    # --- TEXTBLOB BAR CHART (Polarity + Subjectivity for whole text) ---
    st.divider()
    st.subheader("📊 TextBlob Overall Sentiment Metrics")
    
    # Calculate overall polarity & subjectivity of full input
    overall_blob = TextBlob(text)
    overall_polarity = overall_blob.sentiment.polarity
    overall_subjectivity = overall_blob.sentiment.subjectivity
    
    # Centered layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig_tb, ax_tb = plt.subplots(figsize=(6, 4))
        sns.barplot(x=['Polarity', 'Subjectivity'], y=[overall_polarity, overall_subjectivity], palette='coolwarm', ax=ax_tb)
        ax_tb.set_ylim(-1, 1)
        ax_tb.set_ylabel("Score")
        ax_tb.set_title("TextBlob Sentiment Scores for Full Text")
        st.pyplot(fig_tb)

    
    # Word Cloud
    st.subheader("☁️ Word Cloud")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        wc = WordCloud(width=600, height=300, background_color='white').generate(' '.join(words))
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.imshow(wc, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)
    
    # PCA Visualization
    st.subheader("📉 PCA Visualization of TF-IDF Features")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tfidf = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf.fit_transform(df['Sentence']).toarray()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_tfidf)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['Sentiment'] = df['Sentiment']
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Sentiment', palette='Set2', ax=ax3)
        ax3.set_title("PCA of TF-IDF Features")
        st.pyplot(fig3)

    st.success("✅ Analysis Complete! Scroll up to view your results.")
else:
    st.info("👆 Please enter some text in the box above to start analysis.")
