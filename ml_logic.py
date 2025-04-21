# ml_logic.py
import streamlit as st
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, pipeline,
    T5ForConditionalGeneration, T5Tokenizer, PreTrainedModel
)
from transformers_interpret import SequenceClassificationExplainer
import logging
import html
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import io
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
SENTIMENT_MODELS = {
    "BERTweet (Default)": "finiteautomata/bertweet-base-sentiment-analysis",
    "DistilBERT (English)": "distilbert-base-uncased-finetuned-sst-2-english",
}
DEFAULT_SENTIMENT_MODEL_KEY = "BERTweet (Default)"
# LLM_EXPLAINER_MODEL_NAME = "google/flan-t5-small" # Keep for loading placeholder, but won't use for generation now
LABEL_MAP = { "POS": "Positive", "NEG": "Negative", "NEU": "Neutral", "POSITIVE": "Positive", "NEGATIVE": "Negative", "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive" }

# --- Model Loaders (Cached) --- Keep as previous version ---
@st.cache_resource
def load_model_and_tokenizer(model_name):
    logger.info(f"Loading model/tokenizer: {model_name}")
    try:
        try: import emoji
        except ImportError: pass
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try: model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except (OSError, ValueError):
            try:
                 model = T5ForConditionalGeneration.from_pretrained(model_name)
                 tokenizer = T5Tokenizer.from_pretrained(model_name)
            except Exception as e2: raise e2
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model {model_name}.")
        return None, None

@st.cache_resource
def load_sentiment_pipeline(model_name, _model, _tokenizer):
    if _model and _tokenizer:
        if isinstance(_model, (AutoModelForSequenceClassification.register.__self__, PreTrainedModel)) and not isinstance(_model, T5ForConditionalGeneration):
             try:
                 classifier = pipeline('sentiment-analysis', model=_model, tokenizer=_tokenizer, device=-1)
                 return classifier
             except Exception as e: return None
        else: return None
    else: return None

@st.cache_resource
def load_explainer(model_name, _model, _tokenizer):
    if _model and _tokenizer:
        try:
            explainer = SequenceClassificationExplainer(_model, _tokenizer)
            return explainer
        except Exception as e: return None
    else: return None

# Keep LLM loading function as a placeholder for future Gemini integration
@st.cache_resource
def load_llm_explainer_components(llm_model_name="google/flan-t5-small"): # Default name doesn't matter much now
    """Loads placeholder LLM components."""
    logger.info(f"Attempting to load placeholder LLM components...")
    # Simulate loading for now, replace with actual Gemini client setup later
    # For the hackathon, we just need something to return so the app doesn't error
    # Return dummy objects or None
    # return "dummy_model", "dummy_tokenizer"
    # Or return None if Gemini client setup will happen differently
    return None, None # Indicate components aren't truly loaded/usable now


# --- Core Functions ---

def get_prediction(text, pipeline_object):
    # Keep as previous working version
    if not pipeline_object: return None
    if not text or not isinstance(text, str): return None
    try:
        results = pipeline_object(text)
        if results and isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            prediction = results[0]
            raw_label = prediction.get('label', 'Unknown')
            score = prediction.get('score', 0.0)
            mapped_label = LABEL_MAP.get(str(raw_label).upper(), str(raw_label))
            return { 'label': mapped_label, 'score': score }
        else: return None
    except Exception as e: return None

def _generate_simple_html_explanation(word_attributions):
    """Generates HTML with intensity colors, dark background, and bubbles."""
    # Keep as previous working version
    if not word_attributions or not isinstance(word_attributions, list): return "<p>Invalid attribution data.</p>"
    container_bg = "#262730"; text_color = "#FAFAFA"
    positive_hue = 210; negative_hue = 0; neutral_hue = 55
    base_saturation = 65; min_lightness = 40; max_lightness = 85
    neutral_threshold = 0.05
    html_parts = []
    valid_scores = [s for _, s in word_attributions if isinstance(s, (int, float))]
    if not valid_scores: return "<p>No valid scores.</p>"
    max_abs_score = max(abs(score) for score in valid_scores) if valid_scores else 1
    if max_abs_score == 0: max_abs_score = 1
    html_parts.append(f"<div style='line-height: 2.0; font-size: 1rem; font-family: sans-serif; padding: 15px; border: 1px solid #444; border-radius: 8px; background-color: {container_bg}; color: {text_color};'>")
    current_pos = 0
    for item in word_attributions:
        if not isinstance(item, (list, tuple)) or len(item) != 2: continue
        word, score = item
        if not isinstance(word, str) or not isinstance(score, (int, float)): continue
        if word in ['[CLS]', '[SEP]', '<cls>', '<sep>', '<s>', '</s>', '[UNK]', '<unk>', '<pad>']: continue
        escaped_word = html.escape(word)
        normalized_abs_score = abs(score) / max_abs_score
        lightness = int(max_lightness - normalized_abs_score * (max_lightness - min_lightness))
        if score > neutral_threshold: hue = positive_hue
        elif score < -neutral_threshold: hue = negative_hue
        else: hue = neutral_hue; lightness = 75
        color = f"hsl({hue}, {base_saturation}%, {lightness}%)"
        bubble_text_color = "#111" if lightness > 65 else "#eee"
        add_space = " " if current_pos > 0 and not word.startswith(("'",".")) else ""
        current_pos += len(word) + len(add_space)
        html_parts.append(f"{add_space}<span title='Score: {score:.3f}' style='background-color: {color}; color: {bubble_text_color}; padding: 0.2em 0.5em; margin: 0.1em; border-radius: 1em; display: inline-block; box-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>{escaped_word}</span>")
    html_parts.append("</div>")
    return "".join(html_parts)

def get_explanation_data(text, explainer_object):
    # Keep as previous working version
    if not explainer_object: return None
    if not text or not isinstance(text, str): return None
    try:
        word_attributions = explainer_object(text)
        if word_attributions is None: return None
        if not isinstance(word_attributions, list) or not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in word_attributions): return None
        return word_attributions
    except Exception as e: return None

def _filter_and_sort_attributions(word_attributions, top_n=15):
    # Keep as previous working version
    if not word_attributions or not isinstance(word_attributions, list): return None
    filtered_attrs = []
    for item in word_attributions:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            word, score = item
            if isinstance(word, str) and isinstance(score, (int, float)):
                if word not in ['[CLS]', '[SEP]', '<cls>', '<sep>', '<s>', '</s>', '[UNK]', '<unk>', '<pad>']:
                    filtered_attrs.append((word, score))
    if not filtered_attrs: return None
    filtered_attrs.sort(key=lambda item: abs(item[1]), reverse=True)
    return filtered_attrs[:top_n]

def generate_explanation_graph(word_attributions, top_n=15):
    """ Generates a Plotly bar chart. """
    # Keep as previous working version
    top_attrs = _filter_and_sort_attributions(word_attributions, top_n)
    if not top_attrs: return None
    try:
        pos_attrs = sorted([(w, s) for w, s in top_attrs if s > 0], key=lambda item: item[1], reverse=True)
        neg_attrs = sorted([(w, s) for w, s in top_attrs if s <= 0], key=lambda item: item[1], reverse=True)
        plot_data = pos_attrs + neg_attrs
        if not plot_data: return None
        words = [item[0] for item in plot_data]
        scores = [item[1] for item in plot_data]
        colors = ['#1f77b4' if s > 0 else '#d62728' for s in scores]
        fig = go.Figure(go.Bar( x=scores, y=words, orientation='h', marker_color=colors, text=[f'{s:.3f}' for s in scores], textposition='outside'))
        fig.update_layout( title=f'Top {len(plot_data)} Influential Words (Bar)', xaxis_title="Attribution Score", yaxis_title="Word", yaxis=dict(autorange="reversed"), margin=dict(l=100, r=20, t=50, b=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#eee"), height=max(400, len(plot_data) * 35) )
        fig.update_xaxes(zerolinecolor='#555', gridcolor='#444'); fig.update_yaxes(zerolinecolor='#555', gridcolor='#444')
        return fig
    except Exception as e: return None

def generate_matplotlib_pie_chart(word_attributions, top_n=15):
    """Generates a Matplotlib PIE chart showing score distribution."""
    # Keep as previous working version
    top_attrs = _filter_and_sort_attributions(word_attributions, top_n)
    if not top_attrs: return None
    try:
        pos_scores = [s for _, s in top_attrs if s > 0]; neg_scores = [s for _, s in top_attrs if s < 0]
        pos_count = len(pos_scores); neg_count = len(neg_scores)
        if pos_count == 0 and neg_count == 0: return None
        labels = []; sizes = []; colors = []
        if pos_count > 0: labels.append(f'Positive ({pos_count})'); sizes.append(pos_count); colors.append('#6baed6') # Blue
        if neg_count > 0: labels.append(f'Negative ({neg_count})'); sizes.append(neg_count); colors.append('#fc8d59') # Orange/Red
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'}, textprops={'color': 'white'})
            ax.axis('equal'); ax.set_title(f'Pos vs. Neg Scores\n(Top {len(top_attrs)} Words)', color='white')
            plt.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', transparent=True); buf.seek(0); plt.close(fig)
            return buf
    except Exception as e: return None

def generate_wordcloud_image(word_attributions):
    """Generates WordCloud."""
    # Keep as previous working version
    if not word_attributions or not isinstance(word_attributions, list): return None
    try:
        frequencies = {item[0]: abs(item[1]) for item in word_attributions if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (int, float)) and item[0] not in STOPWORDS and item[0] not in ['[CLS]', '[SEP]', '<cls>', '<sep>', '<s>', '</s>', '[UNK]', '<unk>', '<pad>'] and abs(item[1]) > 0.01}
        if not frequencies: return None
        wc = WordCloud(width=800, height=400, background_color='rgba(40, 40, 40, 192)', colormap='viridis', max_words=60, stopwords=STOPWORDS, prefer_horizontal=0.9)
        wc.generate_from_frequencies(frequencies); buf = io.BytesIO(); wc.to_image().save(buf, format='png'); buf.seek(0)
        return buf
    except Exception as e: return None

# Remove the actual LLM generation logic for now, just return a placeholder
def generate_llm_explanation(llm_model, llm_tokenizer, sentiment_label, word_attributions, original_text, top_n=5):
    """Returns placeholder text for LLM explanation."""
    if not sentiment_label: return "Awaiting prediction..."
    # Placeholder text indicating future Gemini integration
    placeholder_text = f"**AI Summary (Future Feature):** An AI like Google Gemini could be integrated here to provide a natural language explanation based on the '{sentiment_label}' prediction and key word contributions."
    return placeholder_text