from dotenv import load_dotenv
load_dotenv()
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from google import genai
from pathlib import Path
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vectors_store"

index=faiss.read_index(str(VECTOR_DIR / "invoice_vectors.index"))
with open(VECTOR_DIR / "invoice_texts.pkl", "rb") as f:
    invoice_texts = pickle.load(f)

embedding_model=SentenceTransformer('all-MiniLM-L6-v2')
query="which invoices are unpaid and high value?"

query_embedding = embedding_model.encode([query])

k=5
distances, indices = index.search(query_embedding, k)

context = "\n\n".join([invoice_texts[i] for i in indices[0]])

prompt = f"""
    you are a financial assistant.
    Use only the invoice data below to answer the question.
    Invoice data:
    {context}
    Question: {query}
    Answer clear and concisely.
    """
response = client.models.generate_content(
    model="models/gemini-flash-lite-latest",
    contents=prompt,
)
print("\nFinal Answer:\n", response.text)
