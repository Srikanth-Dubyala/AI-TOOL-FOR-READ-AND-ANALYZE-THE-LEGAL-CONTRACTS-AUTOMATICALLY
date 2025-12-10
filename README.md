# AI-TOOL-FOR-READ-AND-ANALYZE-THE-LEGAL-CONTRACTS-AUTOMATICALLY
# MODULUS REQURIED TO IMPORT 
# 1.pip install langchain
1.from langchain_community.document_loaders import PyPDFLoader,TextLoader,UnstructuredWordDocumentLoader:  To Load different types of documents of different structure.
2.from langchain_text_splitters import RecursiveCharacterTextSplitter:To split the data ,semantic chunks
# 2.pip install Sentence Transformers
a.from sentence_transformers import SentenceTransformer: To convert the text to embeddings(to numbers)
# 3.pip install huggingface-hub
a.from huggingface_hub import InferenceClient:Here Inference Client act as a interface which send input data to a model and recieves prediction here API'S  works for as authentication
# 4.Here LLM is used Qwen/Qwen2.5-7B-Instruct developed by alibaba it is an cloud open source llm .It has fine tuned to follow human instructions and chat prompts effectively excellent for converstion
# 5.Here I have 4 different AI AGENT for finance,legal,contract,operation
