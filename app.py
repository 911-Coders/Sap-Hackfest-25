# app.py
import streamlit as st

# --- Page Config (Must be first Streamlit command) ---
st.set_page_config( page_title="Explainify!", page_icon="💡", layout="wide", initial_sidebar_state="expanded" )

# --- Imports ---
from ml_logic import (
    SENTIMENT_MODELS, DEFAULT_SENTIMENT_MODEL_KEY,
    load_model_and_tokenizer, load_sentiment_pipeline, load_explainer,
    load_llm_explainer_components, # Now configures Gemini
    get_prediction, get_explanation_data,
    _generate_simple_html_explanation,
    generate_explanation_graph,
    generate_matplotlib_pie_chart,
    generate_wordcloud_image,
    generate_llm_explanation # Now uses Gemini model object
)
import time

# --- Initialize Session State ---
if 'current_model_key' not in st.session_state:
    st.session_state['current_model_key'] = DEFAULT_SENTIMENT_MODEL_KEY
if 'results_cache' not in st.session_state:
    st.session_state['results_cache'] = {}

# --- Load Models & LLM (Cached) ---
@st.cache_data
def load_ai_components(model_name):
    _model, _tokenizer = load_model_and_tokenizer(model_name)
    _pipeline_obj, _explainer_obj = None, None
    if _model and _tokenizer:
        _pipeline_obj = load_sentiment_pipeline(model_name, _model, _tokenizer)
        _explainer_obj = load_explainer(model_name, _model, _tokenizer)
    return _pipeline_obj, _explainer_obj

# Modified: Load LLM now returns the configured Gemini model object
@st.cache_resource # Use cache_resource for client objects
def load_llm():
    """Loads and returns the configured Gemini model object."""
    return load_llm_explainer_components() # This function now handles API key check

# Load components based on session state
current_model_name = SENTIMENT_MODELS[st.session_state.current_model_key]
with st.spinner(f"Loading AI: '{st.session_state.current_model_key}'..."):
    sentiment_pipeline, cls_explainer = load_ai_components(current_model_name)

# Load the Gemini model object
# Spinner not strictly needed unless initial config is slow
gemini_model = load_llm() # This now returns the configured model or None

# --- Sidebar Content --- (Keep as previous version)
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    try: st.image("logo.png", width=180)
    except FileNotFoundError: st.markdown("<h1 style='text-align: center; font-size: 52px;'>💡</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("About Explainify!")
    st.info("Analyze text sentiment using AI models and understand *why* a decision was made through various explanations.")
    st.markdown("---")
    st.header("Team Roles")
    st.markdown("- **TATHAGATA S.** : Lead, Full Stack\n- **TAMAGHNA P.** : Backend\n- **SAYAN K. R.** : Frontend\n- **ABDUL S.** : UI Lead, PPT")
    st.markdown("---")
    st.caption("Hackathon Project")

# --- Main Page Content ---

# Top row: Title and Model Selection
top_col1, top_col2 = st.columns([2, 1])
with top_col1:
    st.title("💡 Explainify!")
    st.markdown(f"Using the **{st.session_state.current_model_key}** model for analysis.")
with top_col2:
    st.markdown(" ")
    selected_model_key = st.selectbox("Choose AI Model:", options=list(SENTIMENT_MODELS.keys()), key='selected_model_key_input', index=list(SENTIMENT_MODELS.keys()).index(st.session_state.current_model_key), label_visibility="collapsed")
    if selected_model_key != st.session_state.current_model_key:
        st.session_state.current_model_key = selected_model_key
        st.session_state.results_cache = {}
        st.rerun()

# --- Input & Prediction Row ---
st.markdown("---")
with st.container():
    input_col, prediction_col = st.columns([5, 6])
    with input_col:
        st.subheader("1. Input Text")
        default_text = "The service was okay, nothing special."
        user_input = st.text_area("Enter text:", default_text, height=200, key="user_input_area")
        analyze_button = st.button("🔍 Analyze & Explain", key="analyze_button", use_container_width=True)
        st.caption("Note: Model interpretations can vary.")
    with prediction_col:
        st.subheader("2. Analysis Result")
        st.markdown("<br>", unsafe_allow_html=True)
        analysis_key = f"{st.session_state.current_model_key}_{user_input}"
        prediction_placeholder = st.empty()

        # Perform analysis on button click
        if analyze_button and user_input:
            prediction_placeholder.empty()
            with st.spinner("🧠 Analyzing & Explaining..."):
                prediction = get_prediction(user_input, sentiment_pipeline)
                explanation_data = None
                llm_explanation = "AI Summary requires analysis."

                if prediction and cls_explainer:
                     explanation_data = get_explanation_data(user_input, cls_explainer)
                     # Call Gemini explanation function - pass the loaded model object
                     if gemini_model and explanation_data: # Check if gemini model loaded AND we have data
                         llm_explanation = generate_llm_explanation(
                             gemini_model, # Pass the configured model object
                             prediction.get('label'),
                             explanation_data,
                             user_input
                         )
                     elif not gemini_model:
                          llm_explanation = "AI Summary generator not available (check API key)."
                     elif not explanation_data:
                          llm_explanation = "Word contributions needed for AI Summary."

                elif prediction and not cls_explainer:
                     llm_explanation = f"{prediction.get('label')} predicted. Detailed explanations unavailable for this model. AI Summary cannot be generated."
                elif not prediction:
                     llm_explanation = "Prediction failed. AI Summary cannot be generated."

                st.session_state.results_cache[analysis_key] = {
                    'prediction': prediction, 'explanation_data': explanation_data,
                    'llm_explanation': llm_explanation, 'analyzed_text': user_input
                }

        # Display prediction if available
        if analysis_key in st.session_state.results_cache:
            results = st.session_state.results_cache[analysis_key]
            prediction = results['prediction']
            if prediction:
                 with prediction_placeholder.container():
                     sentiment_label = prediction['label']
                     st.markdown(f"<div style='text-align: center; margin-bottom: 10px;'>", unsafe_allow_html=True)
                     if sentiment_label == 'Positive': st.success(f"**{sentiment_label}**")
                     elif sentiment_label == 'Negative': st.error(f"**{sentiment_label}**")
                     else: st.info(f"**{sentiment_label}**")
                     st.markdown(f"</div>", unsafe_allow_html=True)
                     st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
                     st.metric(label="Confidence Score", value=f"{prediction['score']:.1%}")
                     st.markdown(f"</div>", unsafe_allow_html=True)
            elif analyze_button:
                 prediction_placeholder.error("Analysis failed to retrieve prediction.")
        elif analyze_button and not user_input:
             prediction_placeholder.warning("⚠️ Please enter text to analyze!")


# --- Explanation Details Section (Below Input/Prediction) ---
st.markdown("---")
st.header("3. Explanation Details")

if analysis_key in st.session_state.results_cache and st.session_state.results_cache[analysis_key]['prediction']:
    results = st.session_state.results_cache[analysis_key]
    explanation_data = results['explanation_data']
    llm_explanation = results['llm_explanation'] # Get generated text or error message

    # Row 1: Summary and Highlighted Text
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.subheader("📝 AI Summary")
        st.info(llm_explanation) # Display the Gemini output or error/placeholder
        # Removed caption about future integration

    with exp_col2:
        st.subheader("Highlighted Text")
        if explanation_data:
            explanation_html = _generate_simple_html_explanation(explanation_data)
            if explanation_html: st.markdown(explanation_html, unsafe_allow_html=True)
            else: st.warning("Could not generate highlights.")
        else: st.warning("Highlight data unavailable.")
        st.caption("Color intensity ≈ importance. Bubble color ≈ sentiment contribution.")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Row 2: Graphs (Keep as before)
    st.subheader("Visual Explanations")
    graph_container_style = "<style>.graph-container{border:1px solid #444; border-radius:8px; padding:15px; background-color:#262730; margin-bottom:15px;}</style>"
    st.markdown(graph_container_style, unsafe_allow_html=True)
    graph_col1, graph_col2, graph_col3 = st.columns(3)
    with graph_col1:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        st.markdown("**Plotly Bar Chart**", help="Shows top words contributing positively (blue) or negatively (red).")
        if explanation_data:
            explanation_fig_plotly = generate_explanation_graph(explanation_data)
            if explanation_fig_plotly: st.plotly_chart(explanation_fig_plotly, use_container_width=True)
            else: st.warning("Graph unavailable.")
        else: st.warning("Data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)
    with graph_col2:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        st.markdown("**Score Distribution (Pie)**", help="Proportion of positive vs. negative top words.")
        if explanation_data:
            explanation_fig_mpl = generate_matplotlib_pie_chart(explanation_data)
            if explanation_fig_mpl: st.image(explanation_fig_mpl)
            else: st.warning("Chart unavailable.")
        else: st.warning("Data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)
    with graph_col3:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        st.markdown("**Word Cloud**", help="Word size indicates impact magnitude.")
        if explanation_data:
            wordcloud_img = generate_wordcloud_image(explanation_data)
            if wordcloud_img: st.image(wordcloud_img)
            else: st.warning("Cloud unavailable.")
        else: st.warning("Data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

elif analyze_button:
    st.warning("Run analysis first or check for errors above to see explanations.")

# --- Footer ---
st.markdown("---")
st.caption("Explainify! - Hackathon Project")