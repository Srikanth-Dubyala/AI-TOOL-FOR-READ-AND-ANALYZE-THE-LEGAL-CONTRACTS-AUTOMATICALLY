#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
from typing import List, TypedDict

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.json import parse_json_markdown

from langgraph.graph import StateGraph, END

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=3072,
    temperature=0.1
)

chat_model = ChatHuggingFace(llm=llm)


# In[4]:


class AgentState(TypedDict):
    chunks: List
    analysis: dict


# In[5]:


def doc_types_and_split(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


# In[6]:



# In[21]:


MASTER_PROMPT = PromptTemplate(
    input_variables=["contract"],
    template="""
You are a contract risk analyzer.

Return ONLY valid JSON.

{{
  "finance": [
    {{
      "clause": "",
      "risk_level": "LOW|MEDIUM|HIGH",
      "reason": "",
      "impact": "",
      "recommendation": ""
    }}
  ],
  "legal": [
    {{
      "clause": "",
      "risk_level": "LOW|MEDIUM|HIGH",
      "reason": "",
      "issue": "",
      "explanation": "",
      "recommendation": ""
    }}
  ],
  "operations": [
    {{
      "clause": "",
      "risk_level": "LOW|MEDIUM|HIGH",
      "reason": "",
      "impact": "",
      "action": ""
    }}
  ],
  "compliance": [
    {{
      "clause": "",
      "risk_level": "LOW|MEDIUM|HIGH",
      "reason": "",
      "violation": "",
      "required_action": ""
    }}
  ]
}}

Rules:
- Do not add explanations
- Do not add markdown
- Output pure JSON only

Contract:
{contract}
"""
)
    
import json
import re


def safe_json_parse(text: str):
    import json, re
    text = re.sub(r"```json|```", "", text)
    
    # Fix bad backslashes that aren't followed by JSON-legal escapes
    text = re.sub(r'\\(?!["\\/bfnrt])', r"\\\\", text)
    # Replace double backslashes
    text = re.sub(r'\\\\', " ", text)

    text = text.strip()
    

    # If it doesn't even end like JSON, it's truncated
    if not text.endswith("}"):
        print("MODEL OUTPUT WAS TRUNCATED")
        print(text[-500:])
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fix bad escapes
        fixed = text.replace("\\", "\\\\")
        try:
            return json.loads(fixed)
        except Exception:
            print(" MODEL RETURNED INVALID JSON:")
            print(text[:1000])
            return None



def analyzer_node(state: AgentState):
    analysis = {}
    for domain in ["finance", "legal", "operations", "compliance"]:
        # Safely get text from chunks
        full_text = "\n".join(c.page_content for c in state.get("chunks", []))
        
        # Get model response
        response = chat_model.invoke(
            MASTER_PROMPT.format(contract=full_text)
        ).content
        
        # Parse JSON from markdown
        parsed = safe_json_parse(response)
        
        # Store domain-specific analysis
        analysis[domain] = parsed.get(domain, [])
    
    return {"analysis": analysis}




# In[34]:


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyzer", analyzer_node)
    graph.set_entry_point("analyzer")
    graph.add_edge("analyzer", END)
    return graph.compile()

app = build_graph()


# In[35]:


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
pc= Pinecone(api_key = os.environ.get("PINECONE_AI_KEY"))
index_name = pc.Index("contract-analysis")
vectorstore = PineconeVectorStore(
    index=index_name,
    embedding=embedding_model
    )
print("vector initlaized successfully")



# In[36]:


def store_analysis_to_pinecone(analysis: dict, contract_id: str):
    texts = []
    metadatas = []
    ids = []

    def add(domain, items):
        for i, item in enumerate(items):
            clause = item.get("clause", "")
            if not clause.strip():
                continue
            texts.append(clause)
            metadatas.append({
                "contract_id": contract_id,
                "domain": domain,
                "risk_level": item.get("risk_level", "UNKNOWN"),
                "data": json.dumps(item)
            })
            ids.append(f"{contract_id}_{domain}_{i}")

    add("finance", analysis.get("finance", []))
    add("legal", analysis.get("legal", []))
    add("operations", analysis.get("operations", []))
    add("compliance", analysis.get("compliance", []))

    if not texts:
        print("Warning: No vectors generated. Model returned empty sections.")
        return

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Stored {len(texts)} vectors in Pinecone.")



# In[37]:


import uuid

def generate_contract_id():
    return str(uuid.uuid4())


# In[40]:


def format_report_for_ui(analysis: dict):
    risk_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def max_risk(items):
        if not items:
            return "LOW"
        return max([i.get("risk_level", "LOW") for i in items], key=lambda x: risk_order.get(x, 1))

    # Overview
    overview = {
        "overall_risk": max_risk([item for domain in analysis.values() for item in domain]),
        "summary": "Contract analysis generated by AI.",
        "finance_risk": max_risk(analysis.get("finance", [])),
        "legal_risk": max_risk(analysis.get("legal", [])),
        "operations_risk": max_risk(analysis.get("operations", [])),
        "compliance_risk": max_risk(analysis.get("compliance", [])),
    }

    def fill_items(domain_name, domain_items):
        for i in domain_items:
            if domain_name == "finance":
                i.setdefault("impact", i.get("impact", "N/A"))
                i.setdefault("recommendation", i.get("recommendation", "N/A"))
            elif domain_name == "legal":
                i.setdefault("issue", i.get("issue", "N/A"))
                i.setdefault("explanation", i.get("explanation", "N/A"))
                i.setdefault("recommendation", i.get("recommendation", "N/A"))
            elif domain_name == "operations":
                i.setdefault("type", i.get("type", "N/A"))
                i.setdefault("impact", i.get("impact", "N/A"))
                i.setdefault("action", i.get("action", "N/A"))
            elif domain_name == "compliance":
                i.setdefault("area", i.get("area", "N/A"))
                i.setdefault("violation", i.get("violation", "N/A"))
                i.setdefault("required_action", i.get("required_action", "N/A"))
        return domain_items

    return {
        "overview": overview,
        "finance": fill_items("finance", analysis.get("finance", [])),
        "legal": fill_items("legal", analysis.get("legal", [])),
        "operations": fill_items("operations", analysis.get("operations", [])),
        "compliance": fill_items("compliance", analysis.get("compliance", []))
    }

def run_contract_analysis(file_path: str):
    # Generate unique contract ID
    contract_id = generate_contract_id()
    
    # Split document into chunks
    chunks = doc_types_and_split(file_path)
    
    # Run backend analysis graph
    result = app.invoke({"chunks": chunks})
    analysis = result["analysis"]
    
    # Store raw analysis in Pinecone
    store_analysis_to_pinecone(analysis, contract_id)
    
    # Format for UI
    report = format_report_for_ui(analysis)
    report["contract_id"] = contract_id
    
    return report


# In[38]:




# In[ ]:








