import json

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser, Document


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    if isinstance(docs, str):
        return docs  # Directly return if already a string
    elif isinstance(docs, list) and all(isinstance(doc, Document) for doc in docs):
        return "\n\n".join(doc.page_content for doc in docs)
    elif isinstance(docs, list):  # Handle unexpected list elements
        return "\n\n".join(str(doc) for doc in docs)
    else:
        raise ValueError(f"Invalid document format: {type(docs)}. Expected list of Documents or string.")


difficulty = st.sidebar.selectbox("Select Quiz Difficulty", ("Easy", "Medium", "Hard"))


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.

    Adjust difficulty: {difficulty}.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

questions_chain = {
    "context": format_docs,
    "difficulty": lambda _: difficulty  # Ensure difficulty is injected at runtime
} | questions_prompt | llm


formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read().decode("utf-8")  # Ensure it's a decoded string
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    if isinstance(docs, list) and all(isinstance(doc, Document) for doc in docs):
        return docs  # ✅ Already correct
    elif isinstance(docs, list):
        return [Document(page_content=str(doc)) for doc in docs]  # Convert to Document objects
    elif isinstance(docs, str):
        return [Document(page_content=docs)]  # Convert single string to Document object
    else:
        raise ValueError("split_file() returned an unsupported type.")




@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, difficulty):
    chain = {
        "context": questions_chain,
        "difficulty": lambda _: difficulty  # Ensure it's dynamically evaluated
    } | formatting_chain | output_parser
    return chain.invoke({"context": _docs, "difficulty": difficulty})




@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)

    if isinstance(docs, list) and all(isinstance(doc, Document) for doc in docs):
        return docs  # ✅ Correct format
    elif isinstance(docs, list):
        return [Document(page_content=str(doc)) for doc in docs]  # Convert list of strings
    elif isinstance(docs, str):
        return [Document(page_content=docs)]  # Convert single string to Document
    else:
        raise ValueError("wiki_search() returned an unsupported type.")


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    if not docs:
        st.error("No valid documents found. Please upload a file or search Wikipedia.")
    else:
        response = run_quiz_chain(docs, difficulty)
    score = 0
    max_score = len(response["questions"])

    with st.form("questions_form"):
        st.write(response)
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                score += 1
            elif value is not None:
                st.error("Wrong!")
        submit_button = st.form_submit_button()
    
    if submit_button:
        st.write(f"Your score: {score}/{max_score}")
        if score == max_score:
            st.balloons()
            st.success("Congratulations! You got a perfect score!")
        else:
            if st.button("Retry Quiz"):
                st.experimental_rerun()