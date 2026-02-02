from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

from rows_to__text import invoice_texts

model=SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")
embeddings = model.encode(invoice_texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)#embeddings are stored in the fiass index

print("Total vectors stored:", index.ntotal)

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vectors_store"
VECTOR_DIR.mkdir(exist_ok=True)
#see tmrw
faiss.write_index(index, str(VECTOR_DIR / "invoice_vectors.index"))

with open(VECTOR_DIR / "invoice_texts.pkl", "wb") as f:
    pickle.dump(invoice_texts, f)

print("Vectors store saved successfully.")  