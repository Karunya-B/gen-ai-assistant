import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vectors_store"

index=faiss.read_index(str(VECTOR_DIR / "invoice_vectors.index"))
with open(VECTOR_DIR / "invoice_texts.pkl", "rb") as f:
    invoice_texts = pickle.load(f)

model=SentenceTransformer('all-MiniLM-L6-v2')
query="high value unpaid voices"

query_embedding = model.encode([query])

k=5
distances, indices = index.search(query_embedding, k)
iii
print("Query:", query)
print("\nTop similar invoices:")\

for i in indices[0]:
    print("\n", invoice_texts[i])
    print("-" * 40)