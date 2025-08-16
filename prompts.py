LLM_EXTRACTION_PROMPT = """You are an expert text analysis agent. Your task is to read the user-provided text and identify all terms that appear in the given gazetteer list.
    [GAZETTEER TERMS]: {gazetteer_list}
    {format_instructions}
    [TEXT TO ANALYZE]: {text_content}"""


AGGREGATION_PROMPT = """You are an expert data analyst. Review the original text and the candidate tags. Based on the text's main themes, select the **top {n}** most relevant and significant tags that best summarize the document.
    {format_instructions}
    [CANDIDATE TAGS]: {tag_list}
    [ORIGINAL TEXT TO ANALYZE]: {text_content}"""