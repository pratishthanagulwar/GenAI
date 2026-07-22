import streamlit as st
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from io import BytesIO
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Optional advanced libs — app degrades gracefully if not installed
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Advanced NLP Text Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# THEME / CUSTOM CSS
# =========================================================
def inject_css(dark_mode: bool):
    if dark_mode:
        bg = "#0E1117"
        card_bg = "#1A1D24"
        text_color = "#F5F5F5"
        accent1 = "#7C4DFF"
        accent2 = "#00E5FF"
        border = "#2A2E37"
    else:
        bg = "#F7F8FA"
        card_bg = "#FFFFFF"
        text_color = "#1A1A1A"
        accent1 = "#4F46E5"
        accent2 = "#06B6D4"
        border = "#E5E7EB"

    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {text_color};
        }}
        .hero {{
            padding: 28px 32px;
            border-radius: 18px;
            background: linear-gradient(135deg, {accent1} 0%, {accent2} 100%);
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }}
        .hero-title {{
            font-size: 2.3rem;
            font-weight: 800;
            color: white;
            margin: 0;
        }}
        .hero-sub {{
            font-size: 1.05rem;
            color: rgba(255,255,255,0.9);
            margin-top: 6px;
        }}
        .hero-date {{
            font-size: 0.9rem;
            color: rgba(255,255,255,0.75);
            margin-top: 10px;
            font-weight: 500;
        }}
        .metric-card {{
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 18px 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}
        .metric-value {{
            font-size: 1.9rem;
            font-weight: 800;
            background: linear-gradient(135deg, {accent1}, {accent2});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: {text_color};
            opacity: 0.7;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .section-card {{
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 16px;
        }}
        div[data-testid="stTabs"] button {{
            font-weight: 600;
        }}
        </style>
    """, unsafe_allow_html=True)


# =========================================================
# RESOURCE LOADING (Cached)
# =========================================================
@st.cache_resource(show_spinner="Loading NLTK resources...")
def download_nltk_resources():
    resources = [
        "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"
    ]
    for resource in resources:
        nltk.download(resource, quiet=True)


@st.cache_resource(show_spinner="Loading SpaCy model...")
def load_spacy_model():
    return spacy.load("en_core_web_sm")


download_nltk_resources()
nlp = load_spacy_model()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=70)
    st.markdown("### 👤 User Profile")
    st.write("**Name:** Pratishtha Nagulwar")
    st.write("**Roll No:** 35")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=False)
    show_explanations = st.checkbox("Show NLP Explanations", value=True)
    top_n_words = st.slider("Top N words in frequency chart", 5, 20, 10)

    st.markdown("---")
    st.markdown("### 📦 Optional Modules")
    st.write(f"{'✅' if TEXTBLOB_AVAILABLE else '❌'} Sentiment Analysis (TextBlob)")
    st.write(f"{'✅' if WORDCLOUD_AVAILABLE else '❌'} Word Cloud")
    if not TEXTBLOB_AVAILABLE or not WORDCLOUD_AVAILABLE:
        st.caption("Run `pip install textblob wordcloud` to unlock these.")

inject_css(dark_mode)

# =========================================================
# HERO HEADER
# =========================================================
current_date = datetime.now().strftime("%B %d, %Y")
st.markdown(f"""
    <div class="hero">
        <p class="hero-title">🧠 Advanced NLP Text Analyzer</p>
        <p class="hero-sub">Comprehensive Natural Language Processing — tokenization, morphology, POS,
        entities, sentiment, and dependency analytics in one place.</p>
        <p class="hero-date">📅 {current_date}</p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT AREA
# =========================================================
default_text = ("Barack Obama was born in Hawaii. He served as the 44th President of the United States. "
                 "Google is a massive tech company based in Mountain View, California.")

st.markdown("#### ✍️ Input Text")
text = st.text_area("", height=150, value=default_text, label_visibility="collapsed",
                     placeholder="Paste or type a paragraph to analyze...")

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_button = st.button("🔍 Analyze Text", use_container_width=True, type="primary")
with col2:
    clear_button = st.button("🗑 Clear", use_container_width=True)

if clear_button:
    st.rerun()

# =========================================================
# ANALYSIS LOGIC
# =========================================================
if analyze_button:
    if text.strip() == "":
        st.error("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Running NLP pipeline..."):
            doc = nlp(text)
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            stop_words = set(stopwords.words("english"))
            clean_words = [w for w in words if w.isalnum()]
            filtered_words = [w for w in clean_words if w.lower() not in stop_words]

        # ---------------- METRICS DASHBOARD ----------------
        m1, m2, m3, m4, m5 = st.columns(5)
        metrics = [
            (m1, len(sentences), "Sentences"),
            (m2, len(words), "Raw Tokens"),
            (m3, len(set(w.lower() for w in filtered_words)), "Unique Words"),
            (m4, len(doc.ents), "Entities Found"),
            (m5, round(len(filtered_words) / max(len(sentences), 1), 1), "Words / Sentence"),
        ]
        for col, value, label in metrics:
            with col:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.write("")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Tokenization",
            "✂️ Stemming & Lemmatization",
            "🏷️ POS & Entities",
            "📊 Analytics & Dependencies",
            "💬 Sentiment & Word Cloud"
        ])

        # --- TAB 1: TOKENIZATION ---
        with tab1:
            st.subheader("1️⃣ Sentence Segmentation")
            for i, sentence in enumerate(sentences, 1):
                st.info(f"**{i}.** {sentence}")

            st.markdown("---")
            st.subheader("2️⃣ Word Tokenization & Cleaning")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Raw Tokens ({len(words)}):**")
                st.write(words)
            with col_b:
                st.write(f"**Without Stopwords ({len(filtered_words)}):**")
                st.write(filtered_words)

        # --- TAB 2: STEMMING & LEMMATIZATION ---
        with tab2:
            st.subheader("3️⃣ Stemming vs. Lemmatization")
            if show_explanations:
                st.caption("*Stemming chops off word endings, while Lemmatization converts words to their dictionary root based on context.*")

            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            morph_data = [{
                "Original Word": w,
                "Stemmed (Porter)": stemmer.stem(w),
                "Lemmatized (WordNet)": lemmatizer.lemmatize(w)
            } for w in filtered_words]

            df_morph = pd.DataFrame(morph_data)
            st.dataframe(df_morph, use_container_width=True)

        # --- TAB 3: POS & NER ---
        with tab3:
            col_c, col_d = st.columns(2)
            with col_c:
                st.subheader("4️⃣ Part-of-Speech Tagging")
                if show_explanations:
                    st.caption("*Assigns grammatical categories (Noun, Verb, Adjective) to each word.*")
                pos_tags = nltk.pos_tag(clean_words)
                df_pos = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
                st.dataframe(df_pos, use_container_width=True, height=300)

            with col_d:
                st.subheader("5️⃣ Named Entity Recognition")
                if show_explanations:
                    st.caption("*Identifies proper nouns like People, Organizations, and Locations.*")
                if len(doc.ents) == 0:
                    st.warning("No Named Entities Found.")
                else:
                    ner_data = [{
                        "Entity": ent.text,
                        "Label": ent.label_,
                        "Description": spacy.explain(ent.label_)
                    } for ent in doc.ents]
                    df_ner = pd.DataFrame(ner_data)
                    st.dataframe(df_ner, use_container_width=True, height=300)

        # --- TAB 4: ANALYTICS & DEPENDENCIES ---
        with tab4:
            col_e, col_f = st.columns(2)
            with col_e:
                st.subheader(f"6️⃣ Word Frequency (Top {top_n_words})")
                word_freq = Counter([w.lower() for w in filtered_words])
                df_freq = pd.DataFrame(word_freq.most_common(top_n_words), columns=["Word", "Count"])
                st.bar_chart(df_freq.set_index("Word"))

            with col_f:
                st.subheader("7️⃣ Dependency Parsing")
                if show_explanations:
                    st.caption("*Shows the syntactic relationships between words.*")
                dep_data = [{
                    "Word": t.text, "Dependency": t.dep_, "Head Word": t.head.text
                } for t in doc if t.is_alpha]
                df_dep = pd.DataFrame(dep_data)
                st.dataframe(df_dep, use_container_width=True, height=300)

            st.markdown("---")
            st.subheader("📥 Export Report")
            csv_data = df_morph.merge(
                df_pos, left_on="Original Word", right_on="Word", how="left"
            )[["Original Word", "Stemmed (Porter)", "Lemmatized (WordNet)", "POS Tag"]]
            st.download_button(
                "Download Analysis as CSV",
                data=csv_data.to_csv(index=False).encode("utf-8"),
                file_name="nlp_analysis_report.csv",
                mime="text/csv"
            )

        # --- TAB 5: SENTIMENT & WORD CLOUD ---
        with tab5:
            col_g, col_h = st.columns(2)

            with col_g:
                st.subheader("8️⃣ Sentiment Analysis")
                if TEXTBLOB_AVAILABLE:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    label = "😊 Positive" if polarity > 0.1 else ("😞 Negative" if polarity < -0.1 else "😐 Neutral")
                    st.metric("Overall Sentiment", label, delta=f"{polarity:+.2f} polarity")
                    st.progress(min(max((polarity + 1) / 2, 0.0), 1.0))
                    st.write(f"**Subjectivity:** {subjectivity:.2f} (0 = objective, 1 = subjective)")
                else:
                    st.warning("Install `textblob` to enable sentiment analysis: `pip install textblob`")

            with col_h:
                st.subheader("9️⃣ Word Cloud")
                if WORDCLOUD_AVAILABLE and filtered_words:
                    wc = WordCloud(width=500, height=350, background_color=None, mode="RGBA",
                                   colormap="plasma").generate(" ".join(filtered_words))
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    fig.patch.set_alpha(0.0)
                    st.pyplot(fig)
                else:
                    st.warning("Install `wordcloud` to enable this view: `pip install wordcloud`")

        st.success("✅ Advanced NLP Analysis Completed Successfully!")