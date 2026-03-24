import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# CONFIG
CHUNK_SIZE = 4   # 3–5 lines
TOP_K = 3
DISTANCE_THRESHOLD = 1.5

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 1. DOCUMENT INGESTION + CHUNKING
def load_and_chunk(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []

    for i in range(0, len(lines), CHUNK_SIZE):
        chunk_text = "".join(lines[i:i+CHUNK_SIZE]).strip()

        if chunk_text:
            chunks.append({
                "id": i // CHUNK_SIZE,
                "text": chunk_text,
                "line_range": (i + 1, min(i + CHUNK_SIZE, len(lines)))
            })

    return chunks



# 2. EMBEDDINGS
def create_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts)
    return embeddings


# 3. FAISS VECTOR STORE
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index



# 4. RETRIEVAL
def retrieve(query, index, chunks):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, TOP_K)

    retrieved_chunks = []
    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks, distances


# 5. ANSWER GENERATION (LLM)
def generate_answer(query, retrieved_chunks, distances):
    context = "\n\n".join([c["text"] for c in retrieved_chunks])

    # Fallback check
    if not context.strip() or distances[0][0] > DISTANCE_THRESHOLD:
        return "I could not find this in the provided document."

    prompt = f"""
        You are a helpful AI assistant.

        Answer the question using ONLY the provided context.
        However, DO NOT copy sentences directly.
        Instead, rephrase and summarize the answer in your own words.
        Give a concise answer in 2-3 sentences.

        If the answer is not present, reply exactly:
        "I could not find this in the provided document."

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content



# MAIN PIPELINE
def main():
    print("Loading and processing document...\n")

    chunks = load_and_chunk("data/document.txt")
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    print("RAG System Ready! Ask questions below.\n")

    while True:
        query = input("Enter your question (or 'exit'): ")

        if query.lower() == "exit":
            break

        retrieved_chunks, distances = retrieve(query, index, chunks)

        print("\n=== Retrieved Context ===")
        for c in retrieved_chunks:
            start, end = c["line_range"]

            print(f"[Chunk {c['id']} | Lines {start}-{end}]")
            print(c["text"])
            print()

        answer = generate_answer(query, retrieved_chunks, distances)

        print("\n=== Answer ===")
        print(answer)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()