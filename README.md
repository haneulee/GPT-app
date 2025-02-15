```
python3 -m venv ./env
```

in the virtual environment

```
source env/bin/activate
pip install -r requirements.txt
```

# RAG

RAG is a model that can be used to retrieve documents from a large corpus. It is based on the retriever-reader framework. The retriever is responsible for selecting a subset of documents from a large corpus, and the reader is responsible for extracting the answer from the selected documents. The retriever is based on a dense retrieval model, which uses a pre-trained transformer model to encode the documents and the query. The reader is based on a pre-trained transformer model that is fine-tuned on a question-answering dataset.
