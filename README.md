# Ensemble AI Tag Extraction Engine

An intelligent web application that uses a multi-method (Gazetteer, spaCy NER, LLM) ensemble approach to extract and rank the most relevant keywords and tags from any given text.

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 🚀 Live Demo

**[https://ensemble-ai-tag-extractor.streamlit.app/]**



Simple keyword extraction is often brittle and misses important context. A basic substring search might find irrelevant matches, while a purely semantic (LLM) search might miss specific, required terminology.

This project solves that problem by using an **ensemble approach**. It combines the strengths of three distinct methods to create a robust and intelligent analysis pipeline:

1.  **Gazetteer Search:** A fast and precise method for finding an exact, predefined list of known terms.
2.  **spaCy NER:** A linguistically-aware method that understands word boundaries (tokens) and uses a custom `EntityRuler` for robust phrase matching.
3.  **LLM Extraction:** A powerful, context-aware method that leverages a large language model to find terms based on semantic understanding.

Finally, a "master" LLM agent aggregates the results from all three methods and ranks them based on their relevance to the document's core themes, providing a final list of high-quality, significant tags.

## ✨ Features

*   **Interactive UI:** Built with Streamlit for a clean, user-friendly experience.
*   **Flexible Inputs:** Analyze text by pasting it directly and uploading a custom gazetteer CSV file.
*   **Parallel Processing:** The core extraction logic is orchestrated by **LangGraph**, which runs all three methods in parallel for efficiency.
*   **Intelligent Ranking:** A final LLM agent synthesizes the results, removing noise and selecting only the most relevant tags.
*   **Rich Visualizations:**
    *   **Text Highlighting:** Final tags are color-coded by category and highlighted directly in the source text.
    *   **spaCy NER View:** An interactive `displacy` view shows exactly what the spaCy `EntityRuler` identified.
    *   **Frequency Chart:** A dynamically generated Matplotlib bar chart shows the frequency of the final ranked tags.

## 🛠️ Tech Stack & Architecture

This project uses a modern AI engineering stack to build a robust and scalable application.

*   **Orchestration:** LangGraph
*   **Web Framework:** Streamlit
*   **LLM Integration:** LangChain, Google Gemini API
*   **NLP:** spaCy (for rule-based NER)
*   **Data Handling:** Pandas
*   **Visualization:** Matplotlib

### Architecture

The workflow is managed by a LangGraph Directed Acyclic Graph (DAG) that ensures a clear and efficient flow of data. The graph fans out to run the three extraction methods in parallel and then converges at a final aggregation node.

![Architecture Diagram](./architecture.jpg)

## ⚙️ Installation & Setup

Follow these steps to run the application locally.

**1. Clone the Repository**
git clone https://github.com/sarim-aliii/Ensemble-AI-Tag-Extractor.git
cd Ensemble-AI-Tag-Extractor

**2. Create a Virtual Environment**
It's highly recommended to use a virtual environment to manage dependencies.
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

**3. Install Dependencies**
This project requires the packages listed in requirements.txt.
pip install -r requirements.txt

**4. Download spaCy Model**
python -m spacy download en_core_web_sm

**5. Set Up Environment Variables**
Create a file named .env in the root of the project and add your Google AI API key.
# .env
GOOGLE_API_KEY="AIzaSy..."


▶️ How to Run
Prepare Data Files: Ensure you have content.txt and gazetteer.csv in the root directory to serve as default inputs.
Run the Streamlit App: Open your terminal in the project's root directory and run:
streamlit run app.py
Your browser should automatically open to the application's URL.


🔮 Future Improvements
More File Types: Add support for uploading and analyzing .pdf or .docx files.
Advanced Analytics: Calculate and display a "sentiment score" for the text based on the categories of the found tags.
Gazetteer Management: Allow users to add, edit, or remove terms from the gazetteer directly in the UI and save their custom version.
Batch Processing: Enable users to upload multiple documents and get aggregated results.
License
This project is licensed under the MIT License - see the LICENSE.md file for details.