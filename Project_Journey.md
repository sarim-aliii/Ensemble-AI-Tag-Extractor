***The Journey of the Ensemble AI Tag Extractor***
This document chronicles the evolution of the AI Tag Extractor project, from a basic Python script to a sophisticated, multi-agent web application. It serves as a narrative of the technical decisions, challenges, and learning process involved in its creation.


Phase 1: The Spark - The Limits of a Simple Search
The project began with a simple, common problem: How can we find a predefined list of keywords in a large block of text?
The initial approach was a straightforward Python script.
Tools: Python, Pandas.
Logic:
Load a "gazetteer" (our custom dictionary) from a CSV file into a pandas DataFrame.
Load the target document into a list of lines.
Use nested for loops to check if any term from the gazetteer existed as a substring within any line of the document.
This worked, but its limitations became immediately obvious:
Brittle: It would incorrectly find the term "art" inside the word "department".
Case-Sensitive: It would miss "Burnout" if the gazetteer only contained "burnout".
No Phrase Recognition: It struggled with multi-word terms like "performance review".
Conclusion: A simple substring search was not intelligent enough. We needed a method that understood the structure of language.


Phase 2: Introducing Linguistic Intelligence with spaCy
To overcome the limitations of the simple search, the next logical step was to use a proper NLP library. spaCy was chosen for its speed and its powerful rule-based matching engine.
Key Component: The EntityRuler.
Logic:
The gazetteer was converted into a series of patterns.
These patterns were used to create a custom EntityRuler that was added to a spaCy pipeline.
Instead of searching for substrings, the text was processed by the pipeline, which used tokenization to understand word boundaries.
This was a major improvement:
Solved the Substring Problem: spaCy correctly identified "performance review" as a two-token entity and no longer made incorrect partial matches.
Richer Output: The system could now extract not just the term, but also the category it belonged to (e.g., Stress_Indicator), which was a much more valuable output.
Introduced Visualization: spaCy's built-in displacy visualizer was used to create beautiful, professional-looking highlights of the found entities, making the results easy to interpret.
Conclusion: Rule-based NLP provided a huge leap in accuracy and sophistication. However, it was still rigid. It couldn't find synonyms or understand the text's deeper context.


Phase 3: The AI Leap - From Searching to Instructing with LLMs
The third phase was a paradigm shift. Instead of writing explicit code to find terms, we would instruct a Large Language Model (LLM) to find them for us.
Tools: LangChain, Google Gemini, Pydantic.
Logic:
A detailed prompt was engineered. This prompt instructed the LLM to act as an "expert text analysis agent."
The prompt was given the full text and the list of gazetteer terms as context.
Crucially, Pydantic was used to define a strict JSON output format. This forced the LLM to return its findings in a clean, machine-readable structure, making the process reliable.
This method unlocked new capabilities:
Semantic Understanding: The LLM could identify terms based on context, not just exact matches.
Flexibility: The prompt could be easily modified to ask for different things (e.g., "find only the most negative terms" or "find synonyms for these terms").
Conclusion: LLM-based extraction was the most powerful and flexible method, but it was also the slowest and had an associated API cost. The ideal solution wouldn't be to choose one method, but to combine the strengths of all three.


Phase 4: The Ensemble - Orchestrating a Multi-Agent System with LangGraph
The project's final form is an ensemble system where all three methods work in parallel. This required a powerful orchestration framework. LangGraph was chosen for its ability to define complex workflows as stateful graphs.
Architecture:
A State Graph was designed, mirroring the project's flowchart.
A START node fans out to the three extraction nodes (gazetteer_extraction, spacy_extraction, llm_extraction), which run in parallel.
The outputs from all three nodes are collected.
An aggregation node takes this combined list of "candidate tags" and uses a final LLM call to act as an expert analyst, ranking the candidates and selecting the "Top N" most relevant ones.
The graph terminates at an END node.
This architecture represents the core of the project's intelligence, leveraging the speed of simple search, the precision of spaCy, and the contextual power of an LLM.


Phase 5: From a Script to a Tool - Building the UI with Streamlit
A powerful backend needs an accessible frontend. The entire LangGraph pipeline was wrapped in an interactive web application using Streamlit.
Key UI Features:
A clean sidebar for user inputs: text area, gazetteer file uploader, and a number input for "Top N".
The main area for displaying a hierarchy of results: the final, ranked tags are shown first, complete with a Matplotlib bar chart for frequency.
Interactive Text Highlighting: Custom HTML and CSS are generated to display the original text with the final tags color-coded by category.
Intermediate Results: Expandable sections show the raw output from each of the three methods, including the beautiful displacy visualization from spaCy.
Performance: st.cache_resource was used to ensure that the heavy LLM and spaCy models are loaded only once, making the app fast and responsive.


Phase 6: The Final Polish - Deployment
The final step was to make the project publicly accessible.
Code Refinement: The code was modularized, moving long prompts into a prompts.py file.
Security: A .gitignore file was created to ensure the .env file containing the API key was never committed to version control.
Deployment: The application was deployed on the Streamlit Community Cloud, using its built-in Secrets management to securely provide the API key to the live application.


This journey transformed a simple idea into a robust, multi-agent AI application with a polished user interface, demonstrating an end-to-end understanding of modern AI engineering principles.