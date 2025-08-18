import pandas as pd
import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from typing import List, Optional
import re
from io import StringIO

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from dotenv import load_dotenv
from prompts import LLM_EXTRACTION_PROMPT, AGGREGATION_PROMPT

load_dotenv()


st.set_page_config(layout="wide", page_title="Ensemble Tag Extractor")
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

@st.cache_resource(show_spinner=False)
def get_spacy_pipeline(gazetteer: pd.DataFrame) -> spacy.language.Language:
    """Creates and caches the spaCy pipeline with a custom entity ruler."""
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    patterns = [{'label': row['Category'], 'pattern': row['Term']} for _, row in gazetteer.iterrows()]
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    return nlp

class ExtractionGraphState(TypedDict):
    content_string: str
    content_lines: List[str]
    gazetteer_df: pd.DataFrame
    n_top_tags: int

    gazetteer_tags: Optional[List[str]]
    spacy_tags: Optional[List[str]]
    llm_tags: Optional[List[str]]

    spacy_viz_html: Optional[str]
    final_tags: Optional[List[str]]


def gazetteer_extraction_node(state: ExtractionGraphState) -> dict:
    lines, gazetteer = state["content_lines"], state["gazetteer_df"]
    tags = set()
    for line in lines:
        for tag in gazetteer['Term']:
            if tag.lower() in line.lower():
                tags.add(tag)
    return {"gazetteer_tags": list(tags)}

def spacy_extraction_node(state: ExtractionGraphState) -> dict:
    content, gazetteer = state["content_string"], state["gazetteer_df"]
    nlp_pipeline = get_spacy_pipeline(gazetteer)
    
    doc = nlp_pipeline(content)
    gazetteer_categories = gazetteer['Category'].unique()
    matched_entities = [ent.text for ent in doc.ents if ent.label_ in gazetteer_categories]
    
    custom_ents_doc = spacy.tokens.Doc(doc.vocab, words=[t.text for t in doc])
    custom_ents_doc.ents = [ent for ent in doc.ents if ent.label_ in gazetteer_categories]
    color_map = {
        "Stress_Indicator": "#FF6961", 
        "Positive_Engagement": "#77DD77", 
        "Work_Life_Balance": "#AEC6CF", 
        "Corporate_Process": "#C3B1E1"
    }
    options = {"ents": gazetteer_categories.tolist(), "colors": color_map}
    html = displacy.render(custom_ents_doc, style="ent", options=options, jupyter=False)
    return {"spacy_tags": list(set(matched_entities)), "spacy_viz_html": html}

class ExtractedEntities(BaseModel):
    found_terms: list[str] = Field(description="A list of unique terms from the gazetteer found in the text.")

def llm_extraction_node(state: ExtractionGraphState) -> dict:
    content, gazetteer_terms = state["content_string"], state["gazetteer_df"]['Term'].tolist()
    llm = get_llm()
    
    parser = PydanticOutputParser(pydantic_object=ExtractedEntities)
    prompt = ChatPromptTemplate.from_template(template=LLM_EXTRACTION_PROMPT, partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"gazetteer_list": ", ".join(gazetteer_terms), "text_content": content})
        return {"llm_tags": result.found_terms}
    except Exception as e:
        st.error(f"LLM Extraction Error: {e}")
        return {"llm_tags": []}

class FinalRankedTags(BaseModel):
    top_n_tags: list[str] = Field(description="The top N most relevant tags.")

def aggregation_node(state: ExtractionGraphState) -> dict:
    content_string, n = state["content_string"], state["n_top_tags"]
    llm = get_llm()
    
    candidate_tags = list(set(state.get("gazetteer_tags", []) + state.get("spacy_tags", []) + state.get("llm_tags", [])))
    if not candidate_tags: return {"final_tags": []}
    
    parser = PydanticOutputParser(pydantic_object=FinalRankedTags)
    prompt = ChatPromptTemplate.from_template(template=AGGREGATION_PROMPT, partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"n": n, "tag_list": ", ".join(candidate_tags), "text_content": content_string})
        return {"final_tags": result.top_n_tags}
    except Exception as e:
        st.error(f"LLM Ranking Error: {e}")
        return {"final_tags": []}

def start_router(state: ExtractionGraphState) -> List[str]:
    return ["gazetteer_extraction", "spacy_extraction", "llm_extraction"]

def build_graph():
    graph = StateGraph(ExtractionGraphState)

    graph.add_node("gazetteer_extraction", gazetteer_extraction_node)
    graph.add_node("spacy_extraction", spacy_extraction_node)
    graph.add_node("llm_extraction", llm_extraction_node)
    graph.add_node("aggregation", aggregation_node)

    graph.set_conditional_entry_point(start_router)

    graph.add_edge("gazetteer_extraction", "aggregation")
    graph.add_edge("spacy_extraction", "aggregation")
    graph.add_edge("llm_extraction", "aggregation")
    
    graph.add_edge("aggregation", END)
    
    return graph.compile()


st.title("📄✨ Ensemble AI Tag Extractor")
st.markdown("This tool uses three different methods to extract keywords from text and then uses an LLM to rank the most relevant ones.")

try:
    with open("content.txt", 'r', encoding='utf-8') as f:
        default_text = f.read()
    with open("gazetteer.csv", 'r', encoding='utf-8') as f:
        default_gazetteer = f.read()
except FileNotFoundError:
    default_text = "Please paste your text here..."
    default_gazetteer = None

with st.sidebar:
    st.header("⚙️ Configuration")
    user_text = st.text_area("1. Paste Text to Analyze", value=default_text, height=300)
    uploaded_gazetteer = st.file_uploader("2. Upload Gazetteer CSV", type="csv")
    top_n = st.number_input("3. Number of Final Tags to Rank", min_value=1, max_value=20, value=5, step=1)
    analysis_button = st.button("🚀 Run Analysis", type="primary")

if analysis_button:
    if not user_text.strip():
        st.warning("Please paste some text to analyze.")
    elif uploaded_gazetteer is None and default_gazetteer is None:
        st.warning("Please upload a gazetteer CSV file.")
    else:
        with st.spinner("🤖 AI is thinking... This may take a moment..."):
            if uploaded_gazetteer:
                gazetteer_df = pd.read_csv(uploaded_gazetteer)
            else:
                gazetteer_df = pd.read_csv(StringIO(default_gazetteer))
            
            graph = build_graph()

            initial_state = {
                "content_string": user_text,
                "content_lines": user_text.splitlines(),
                "gazetteer_df": gazetteer_df,
                "n_top_tags": top_n,
            }
            
            final_state = graph.invoke(initial_state)

            st.session_state.results = final_state
            st.session_state.user_text = user_text
            st.session_state.gazetteer_df = gazetteer_df

def highlight_text(content: str, final_tags: list, gazetteer_df: pd.DataFrame) -> str:
    color_map = {
        "Stress_Indicator": "rgba(255, 127, 127, 0.5)", 
        "Positive_Engagement": "rgba(144, 238, 144, 0.6)", 
        "Work_Life_Balance": "rgba(173, 216, 230, 0.6)", 
        "Corporate_Process": "rgba(221, 160, 221, 0.5)", 
        "default": "rgba(255, 255, 0, 0.5)"
    }

    tag_to_category = gazetteer_df.set_index('Term')['Category'].to_dict()
    sorted_tags = sorted(final_tags, key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(tag) for tag in sorted_tags) + r')\b'

    def replace_func(match):
        matched_text = match.group(0)
        category = tag_to_category.get(matched_text, "default")
        color = color_map.get(category, color_map["default"])
        return f'<span style="background-color: {color}; padding: 2px 4px; margin: 0 2px; border-radius: 4px; line-height: 1.6;">{matched_text}</span>'

    highlighted_content = re.sub(pattern, replace_func, content, flags=re.IGNORECASE)
    legend_html = "<div style='margin-bottom: 20px;'><b>Legend:</b> "
    seen_categories = {tag_to_category.get(tag) for tag in final_tags if tag_to_category.get(tag)}
    
    for category in sorted(seen_categories):
        color = color_map.get(category, color_map["default"])
        legend_html += f'<span style="background-color: {color}; padding: 2px 6px; margin: 0 5px; border-radius: 4px;">{category.replace("_", " ")}</span>'
    legend_html += "</div>"
    return legend_html + highlighted_content.replace("\n", "<br>")

if 'results' in st.session_state:
    # (This entire UI display block is unchanged)
    results, user_text, gazetteer_df = st.session_state.results, st.session_state.user_text, st.session_state.gazetteer_df
    final_tags = results.get("final_tags", [])
    st.header("✨ Highlighted Text Analysis")
    if not final_tags:
        st.info("No tags were found to highlight in the text.")
        st.text_area("Original Text", user_text, height=300)
    else:
        highlighted_html = highlight_text(user_text, final_tags, gazetteer_df)
        st.markdown(highlighted_html, unsafe_allow_html=True)
    st.divider()
    st.header("🏆 Final Ranked Tags")
    if not final_tags:
        st.info("No relevant tags were found by the final ranking model.")
    else:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown(f"**Top {len(final_tags)} Most Relevant Tags:**")
            for i, tag in enumerate(final_tags, 1):
                st.success(f"**{i}. {tag}**")
        with col2:
            st.markdown("**Tag Frequency in Text:**")
            tag_counts = {tag: user_text.lower().count(tag.lower()) for tag in final_tags}
            fig, ax = plt.subplots()
            ax.barh(list(tag_counts.keys()), list(tag_counts.values()), color='skyblue')
            ax.set_xlabel('Frequency Count')
            ax.set_title('Frequency of Final Tags')
            plt.tight_layout()
            st.pyplot(fig)
    st.divider()
    st.header("🕵️‍♂️ Intermediate Results from Each Method")
    with st.expander("1. Gazetteer (Simple Search) Results"):
        st.write(results.get("gazetteer_tags", []))
    with st.expander("2. spaCy NER Results", expanded=True):
        st.subheader("Visualization of Found Entities")
        spacy_html = results.get("spacy_viz_html")
        if spacy_html:
            components.html(spacy_html, height=400, scrolling=True)
        else:
            st.info("No entities were found by the spaCy method.")
        st.subheader("List of Found Tags")
        st.write(results.get("spacy_tags", []))
    with st.expander("3. LLM Extraction Results"):
        st.write(results.get("llm_tags", []))