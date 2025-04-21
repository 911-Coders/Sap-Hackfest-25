# app.py
import streamlit as st

# --- Page Config (Must be first Streamlit command) ---
st.set_page_config(
    page_title="Explainify!",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Imports --- (Place after set_page_config)
from ml_logic import (
    SENTIMENT_MODELS, DEFAULT_SENTIMENT_MODEL_KEY,
    load_model_and_tokenizer, load_sentiment_pipeline, load_explainer,
    load_llm_explainer_components, # Still load placeholder
    get_prediction, get_explanation_data,
    _generate_simple_html_explanation,
    generate_explanation_graph,
    generate_matplotlib_pie_chart,
    generate_wordcloud_image,
    generate_llm_explanation # Now returns placeholder
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

@st.cache_data
def load_llm(): # Placeholder loader
    return load_llm_explainer_components()

# Load components based on session state
current_model_name = SENTIMENT_MODELS[st.session_state.current_model_key]
with st.spinner(f"Loading AI: '{st.session_state.current_model_key}'..."):
    sentiment_pipeline, cls_explainer = load_ai_components(current_model_name)
# with st.spinner("Loading Summary Generator..."): # No need to show spinner for placeholder
llm_model, llm_tokenizer = load_llm() # Call placeholder loader

# --- Sidebar Content ---
with st.sidebar:
    # Logo with padding
    st.markdown("<br>", unsafe_allow_html=True) # Top padding
    try:
        st.image("logo.png", width=180) # Centered by default in sidebar column
    except FileNotFoundError:
        st.markdown("<h1 style='text-align: center; font-size: 52px;'>💡</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) # Bottom padding
    st.markdown("---")

    # Project Info
    st.header("About Explainify!")
    st.info(
        """
        Analyze text sentiment using AI models and understand *why* a decision
        was made through various visual and text-based explanations.
        """
    )
    st.markdown("---")
    st.header("Team Roles")
    st.markdown(
        """
        - **TATHAGATA S.** : Lead, Full Stack
        - **TAMAGHNA P.** : Backend
        - **SAYAN K. R.** : Frontend
        - **ABDUL S.** : UI Lead, PPT
        """
    )
    st.markdown("---")
    st.caption("Hackathon Project")

# --- Main Page Content ---

# Top row: Title and Model Selection
top_col1, top_col2 = st.columns([2, 1]) # Title gets more space
with top_col1:
    st.title("💡 Explainify!")
    st.markdown(f"Using the **{st.session_state.current_model_key}** model for analysis.")

with top_col2:
    st.markdown(" ") # Vertical alignment helper
    selected_model_key = st.selectbox(
        "Choose AI Model:", options=list(SENTIMENT_MODELS.keys()),
        key='selected_model_key_input',
        index=list(SENTIMENT_MODELS.keys()).index(st.session_state.current_model_key),
        label_visibility="collapsed"
    )
    if selected_model_key != st.session_state.current_model_key:
        st.session_state.current_model_key = selected_model_key
        st.session_state.results_cache = {}
        st.rerun()

# --- Input & Prediction Row ---
st.markdown("---")
# Use st.container() for better padding control if needed
with st.container():
    input_col, prediction_col = st.columns([5, 6]) # Give slightly more space to results

    with input_col:
        st.subheader("1. Input Text")
        default_text = "The service was okay, nothing special."
        user_input = st.text_area("Enter text:", default_text, height=200, key="user_input_area",
                                  help="Paste or type the text to analyze.")
        analyze_button = st.button("🔍 Analyze & Explain", key="analyze_button", use_container_width=True)
        st.caption("Note: Model interpretations can vary.")

    with prediction_col:
        st.subheader("2. Analysis Result")
        # Add some vertical space before the result
        st.markdown("<br>", unsafe_allow_html=True)
        analysis_key = f"{st.session_state.current_model_key}_{user_input}"
        prediction_placeholder = st.empty() # Placeholder for prediction results

        # Perform analysis on button click
        if analyze_button and user_input:
            prediction_placeholder.empty() # Clear previous
            with st.spinner("🧠 Analyzing..."):
                prediction = get_prediction(user_input, sentiment_pipeline)
                explanation_data = None
                llm_explanation = generate_llm_explanation(None, None, None, None, None) # Get placeholder initially

                if prediction and cls_explainer:
                     explanation_data = get_explanation_data(user_input, cls_explainer)
                     # Get placeholder text using prediction label
                     llm_explanation = generate_llm_explanation(None, None, prediction.get('label'), None, None)
                elif prediction and not cls_explainer:
                     llm_explanation = f"{prediction.get('label')} predicted. Detailed explanations unavailable for this model."
                elif not prediction:
                     llm_explanation = "Prediction failed."

                st.session_state.results_cache[analysis_key] = {
                    'prediction': prediction, 'explanation_data': explanation_data,
                    'llm_explanation': llm_explanation, # Store placeholder
                    'analyzed_text': user_input
                }

        # Display prediction if available
        if analysis_key in st.session_state.results_cache:
            results = st.session_state.results_cache[analysis_key]
            prediction = results['prediction']
            if prediction:
                 with prediction_placeholder.container():
                     # Display label centered
                     sentiment_label = prediction['label']
                     st.markdown(f"<div style='text-align: center; margin-bottom: 10px;'>", unsafe_allow_html=True) # Center alignment div
                     if sentiment_label == 'Positive': st.success(f"**{sentiment_label}**")
                     elif sentiment_label == 'Negative': st.error(f"**{sentiment_label}**")
                     else: st.info(f"**{sentiment_label}**")
                     st.markdown(f"</div>", unsafe_allow_html=True)

                     # Display confidence below, centered
                     st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
                     st.metric(label="Confidence Score", value=f"{prediction['score']:.1%}")
                     st.markdown(f"</div>", unsafe_allow_html=True)

            elif analyze_button: # If analysis ran but failed
                 prediction_placeholder.error("Analysis failed. Could not retrieve prediction.")
        elif analyze_button and not user_input:
             prediction_placeholder.warning("⚠️ Please enter text to analyze!")


# --- Explanation Details Section (Below Input/Prediction) ---
st.markdown("---") # Separator
st.header("3. Explanation Details")

# Display explanations only if analysis was successful
if analysis_key in st.session_state.results_cache and st.session_state.results_cache[analysis_key]['prediction']:
    results = st.session_state.results_cache[analysis_key]
    explanation_data = results['explanation_data']
    llm_explanation = results['llm_explanation'] # Get placeholder text

    # Row 1: Summary Placeholder and Highlighted Text
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.subheader("📝 AI Summary")
        st.warning(llm_explanation) # Display the placeholder text as a warning/notice
        st.caption("(Future integration with Google Gemini API)")

    with exp_col2:
        st.subheader("Highlighted Text")
        if explanation_data:
            explanation_html = _generate_simple_html_explanation(explanation_data)
            if explanation_html: st.markdown(explanation_html, unsafe_allow_html=True)
            else: st.warning("Could not generate highlights.")
        else: st.warning("Highlight data unavailable.")
        st.caption("Color intensity ≈ importance. Bubble color ≈ sentiment contribution.")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Row 2: Graphs
    st.subheader("Visual Explanations")
    # Add border/background to graph columns for visual grouping
    graph_container_style = """
    <style>
        .graph-container {
            border: 1px solid #444;
            border-radius: 8px;
            padding: 15px;
            background-color: #262730; /* Match highlight background */
            margin-bottom: 15px; /* Space below each graph container */
        }
    </style>
    """
    st.markdown(graph_container_style, unsafe_allow_html=True)

    graph_col1, graph_col2, graph_col3 = st.columns(3)
    with graph_col1:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        st.markdown("**Plotly Bar Chart**", help="Shows top words contributing positively (blue) or negatively (red) to the sentiment.")
        if explanation_data:
            explanation_fig_plotly = generate_explanation_graph(explanation_data)
            if explanation_fig_plotly: st.plotly_chart(explanation_fig_plotly, use_container_width=True)
            else: st.warning("Graph unavailable.")
        else: st.warning("Data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    with graph_col2:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        st.markdown("**Score Distribution (Pie)**", help="Shows the proportion of top contributing words that were positive vs. negative.")
        if explanation_data:
            explanation_fig_mpl = generate_matplotlib_pie_chart(explanation_data)
            if explanation_fig_mpl: st.image(explanation_fig_mpl)
            else: st.warning("Chart unavailable.")
        else: st.warning("Data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    with graph_col3:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        st.markdown("**Word Cloud**", help="Visualizes word importance. Larger words had a stronger impact (positive or negative).")
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