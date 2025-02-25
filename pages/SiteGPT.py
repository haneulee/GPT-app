from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

CLOUDFLARE_SITEMAPS = {
    "AI Gateway": "https://developers.cloudflare.com/ai-gateway/sitemap.xml",
    "Vectorize": "https://developers.cloudflare.com/vectorize/sitemap.xml",
    "Workers AI": "https://developers.cloudflare.com/workers-ai/sitemap.xml",
}

st.set_page_config(page_title="Cloudflare SiteGPT", page_icon="☁️")

st.sidebar.title("Cloudflare SiteGPT")
st.sidebar.markdown("""
### Select a Cloudflare product
Ask questions about Cloudflare AI services.

[GitHub Repository](https://github.com/haneulee/GPT-app)
""")

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(url, parsing_function=parse_page)
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=api_key))
    return vector_store.as_retriever()

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return soup.get_text().replace("\n", " ").replace("\xa0", " ")

llm = ChatOpenAI(temperature=0.1, openai_api_key=api_key)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
    
    Then, give a score to the answer between 0 and 5.
    
    If the answer answers the user question the score should be high, else it should be low.
    
    Make sure to always include the answer's score even if it's 0.
    
    Context: {context}
    
    Examples:
    
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
    
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
    
    Your turn!
    
    Question: {question}
    """
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata.get("lastmod", "Unknown"),
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        Use ONLY the following pre-existing answers to answer the user's question.
        
        Use the answers that have the highest score (more helpful) and favor the most recent ones.
        
        Cite sources and return the sources of the answers as they are, do not change them.
        
        Answers: {answers}
        """),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})

selected_product = st.sidebar.selectbox("Select a product", list(CLOUDFLARE_SITEMAPS.keys()))

if api_key:
    retriever = load_website(CLOUDFLARE_SITEMAPS[selected_product])
    query = st.text_input("Ask a question about Cloudflare's AI services")
    if query:
        chain = (
            {"docs": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$"))
else:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
