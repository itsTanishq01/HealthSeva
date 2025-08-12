import os
import json
import pandas as pd
import numpy as np
import time
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from groq import Groq

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

groqApiKey = os.getenv("GROQ_API_KEY")
pineconeApiKey = os.getenv("PINECONE_API_KEY")

if not groqApiKey or not pineconeApiKey:
    raise ValueError("Missing required API keys. Please set GROQ_API_KEY and PINECONE_API_KEY environment variables.")

client = Groq(api_key=groqApiKey)
chatModel = "llama3-8b-8192"
indexName = "mental-health-chat"

with open("MESC_structured.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for dialogue in data:
    for utt in dialogue["Utterances"]:
        rows.append({
            "speaker": utt["Speaker"],
            "text": utt["Utterance"],
            "emotion": utt["Emotion"],
            "strategy": utt["Strategy"]
        })
dataFrame = pd.DataFrame(rows)

def getLocalEmbedding(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    encodedInput = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        modelOutput = model(**encodedInput)
    
    embeddings = modelOutput.last_hidden_state.mean(dim=1)
    
    return [emb.numpy().tolist() for emb in embeddings]

pineconeClient = Pinecone(api_key=pineconeApiKey)

if indexName in [index["name"] for index in pineconeClient.list_indexes()]:
    existingIndex = pineconeClient.Index(indexName)
    indexInfo = pineconeClient.describe_index(indexName)
    expectedDim = len(getLocalEmbedding(["test"])[0])
    
    if indexInfo.dimension != expectedDim:
        print(f"Deleting existing index with wrong dimensions ({indexInfo.dimension} vs {expectedDim})")
        pineconeClient.delete_index(indexName)

if indexName not in [index["name"] for index in pineconeClient.list_indexes()]:
    print("Creating new Pinecone index...")
    pineconeClient.create_index(
        name=indexName,
        dimension=len(getLocalEmbedding(["test"])[0]),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Waiting for index to be ready...")
    time.sleep(10)

vectorIndex = pineconeClient.Index(indexName)

try:
    stats = vectorIndex.describe_index_stats()
    print(f"Index stats: {stats.get('total_vector_count', 0)} vectors")
except Exception as e:
    print(f"Warning: Could not get index stats: {e}")

if vectorIndex.describe_index_stats().get("total_vector_count", 0) == 0:
    print("Uploading embeddings to Pinecone...")
    print(f"Total embeddings to process: {len(dataFrame)}")
    
    batchSize = 100
    total = len(dataFrame)
    
    for batchStart in range(0, total, batchSize):
        batchEnd = min(batchStart + batchSize, total)
        vectors = []
        
        for i in range(batchStart, batchEnd):
            row = dataFrame.iloc[i]
            emb = getLocalEmbedding([row["text"]])[0]
            vectors.append((str(i), emb, {"text": row["text"]}))
        
        vectorIndex.upsert(vectors)
        
        progress = (batchEnd / total) * 100
        print(f"Progress: {progress:.1f}% ({batchEnd}/{total})")
    
    print("Upload complete.")

def getContext(userInput, topK=3):
    try:
        print(f"Getting embedding for: '{userInput[:50]}...'")
        queryEmb = getLocalEmbedding([userInput])[0]
        print(f"Embedding dimension: {len(queryEmb)}")
        
        maxRetries = 3
        for attempt in range(maxRetries):
            try:
                print(f"Querying Pinecone (attempt {attempt + 1})...")
                results = vectorIndex.query(vector=queryEmb, top_k=topK, include_metadata=True)
                print(f"Found {len(results['matches'])} matches")
                context = "\n".join(match["metadata"]["text"] for match in results["matches"])
                print(f"Context length: {len(context)} characters")
                return context
            except Exception as e:
                print(f"Pinecone query attempt {attempt + 1} failed: {e}")
                if attempt < maxRetries - 1:
                    time.sleep(2)
                else:
                    raise e
                    
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return "I'm having trouble accessing my knowledge base right now. Let me try to help you anyway."

def queryGroq(prompt):
    maxRetries = 3
    baseDelay = 5
    
    for attempt in range(maxRetries):
        try:
            print(f"Attempting Groq API call (attempt {attempt + 1}) with model: {chatModel}")
            
            chatCompletion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=chatModel,
                temperature=0.7,
                max_tokens=500
            )
            
            print(f"Groq API call successful")
            return chatCompletion.choices[0].message.content
            
        except Exception as e:
            print(f"Groq API error: {e}")
            if attempt < maxRetries - 1:
                delay = baseDelay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Max retries reached. Last error: {e}")
    
    return "I understand you're reaching out. I'm currently experiencing high demand, but I want you to know that your feelings are valid and it's okay to seek support. Please try again in a few minutes, or if you're in crisis, consider contacting a mental health professional or crisis helpline."

def getChatbotResponse(userInput):
    try:
        context = getContext(userInput)
        prompt = f"""
You are a mental health assistant. Respond empathetically.
Here are relevant past examples from therapy dialogues:

{context}

User: {userInput}
Assistant:"""
        return queryGroq(prompt)
    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later."

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

app = FastAPI(
    title="Mental Health Chatbot API",
    description="A mental health assistant chatbot using MESC therapy dialogue data",
    version="1.0.0"
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reply = getChatbotResponse(request.message)
    return ChatResponse(response=reply)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Mental Health Chatbot API", 
        "docs": "/docs",
        "chat_endpoint": "/chat"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
