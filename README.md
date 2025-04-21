# Sap-Hackfest-25
Hackfest solve for SAP org Hackathon on 22nd April 2025.

# ✨ Explainify! ✨

<p align="center">
  <img src="logo.png" alt="Explainify Logo" width="200"/>
</p>

**Making AI Sentiment Analysis Transparent and Understandable.**

Explainify is an interactive web application built for the SAP Hackfest '25. It allows users to input text, get sentiment predictions from different AI models, and explore various explanations to understand *why* the model reached its conclusion.

---

## 🚀 Key Features

*   **Sentiment Prediction:** Classifies text into Positive, Negative, or Neutral categories.
*   **📊 Model Selection:** Choose between different pre-trained sentiment analysis models (e.g., BERTweet, DistilBERT) via a simple dropdown.
*   **💬 Multiple Explanation Views:**
    *   **📝 AI Summary (Placeholder):** Shows where a future Gemini-powered natural language summary will appear.
    *   **Highlighted Text:** Displays the input text with words colored and styled (intensity + bubbles) based on their contribution to the predicted sentiment (Blue≈Positive, Red≈Negative, Yellow≈Neutral).
    *   **📊 Importance Bar Chart (Plotly):** Interactive horizontal bar chart showing the top words influencing the decision and their attribution scores.
    *   **📊 Score Distribution Pie Chart (Matplotlib):** Visualizes the proportion of positive vs. negative contributing words among the most influential ones.
    *   **☁️ Word Cloud:** Displays influential words, with size indicating the magnitude of their contribution.
*   **💡 Interactive UI:** Built with Streamlit for a user-friendly experience.

---

## 🛠️ Technology Stack

*   **Frontend:** Streamlit
*   **Backend & ML:** Python 3
*   **Core ML Libraries:**
    *   Transformers (Hugging Face) - For loading models & tokenizers
    *   PyTorch (or TensorFlow, depending on backend) - Model execution
    *   transformers-interpret - For generating word attributions
*   **Visualization:**
    *   Plotly - Interactive charts
    *   Matplotlib - Static charts
    *   WordCloud - Word cloud generation
*   **Environment Management:** `venv`

---

## ⚙️ Setup and Installation

1.  **Prerequisites:**
    *   Git installed ([https://git-scm.com/](https://git-scm.com/))
    *   Python 3 installed (Recommended: 3.9+)
    *   Access to a terminal or command prompt.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/911-Coders/Sap-Hackfest-25.git
    cd Sap-Hackfest-25
    ```

3.  **Run the Setup Script (Windows):**
    *   Navigate to the project directory in your Command Prompt or PowerShell.
    *   Execute the `run.bat` script. This will automatically:
        *   Create a Python virtual environment (`venv`).
        *   Install all required dependencies from `requirements.txt`.
    ```bash
    .\run.bat
    ```
    *(Note: On first run, downloading models and dependencies might take some time).*

---

## ▶️ Running the Application

*   Simply execute the `run.bat` script from the project's root directory:
    ```bash
    .\run.bat
    ```
*   The script will handle environment activation and start the Streamlit server.
*   The application should automatically open in your default web browser (usually at `http://localhost:8501`).
*   To stop the application, go back to the terminal where `run.bat` is running and press `Ctrl + C`.

---

## 📁 Project Structure
```Sap-Hackfest-25/
│
├── app.py # Main Streamlit application script
├── ml_logic.py # Core ML/backend functions
├── requirements.txt # Python dependencies
├── run.bat # Windows setup & run script
├── logo.png 
├── .gitignore
└── README.md # This file
```
---

## 🧑‍💻 Team: 911 Coders

*   **TATHAGATA S.** : Team Lead, Full Stack
*   **TAMAGHNA P.** : Backend
*   **SAYAN K. R.** : Frontend
*   **ABDUL S.** : UI Lead, PPT

---

## 🔮 Future Work

*   **Integrate Google Gemini API:** Replace the placeholder AI summary with actual explanations generated via the Gemini API.
*   **Add More Models:** Expand the selection of sentiment analysis models.
*   **Refine Explanations:** Improve compatibility checks for explainers with different model types.
*   **UI/UX Enhancements:** Further polish the user interface and experience.
*   **Error Handling:** Add more robust error handling for edge cases (e.g., very long inputs, unusual characters).

---

## 📄 License

LICENSED UNDER 
MIT License

Copyright (c) 2025 911 Coders

---

Built with ❤️ for SAP Hackfest '25!

