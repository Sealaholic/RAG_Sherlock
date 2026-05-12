from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

class RagSystem:
    def __init__(self, api_key: str = None, docs_dir: str = 'docs'):
        self.api_key = api_key
        self.docs_dir = Path(docs_dir)
        #  Document storage
        self.documents = []
        self.doc_chunks = []
        self.vectorizer = None
        self.doc_vectors = None

        # Load stopwords
        # self.stopwords = self.load_stopwords()

        #. Cache documents
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.chunks_cache = self.cache_dir / "doc_chunks.pkl"
        self.vectors_cache = self.cache_dir / "doc_vectors.pkl"

        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.client = OpenAI(
            api_key=self.api_key
)

    def initialize(self):
        print("=== Book Searching Initializing ===")
        self.load_docs()
        self.preprocess_documents()
        self.build_vector_index()
        print("\nInitialization complete")

    def load_docs(self):
        """Load documents"""
        print("\nLoading...")
        for file_path in self.docs_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.documents.append({
                            'filename': file_path,
                            'content': content,
                            'path': str(file_path)
                        })
            except Exception as e:
                print(f"\nError occurs when loading {file_path}: {e}")
        # print(f"Total {len(self.documents)}")

    def split_text_chunks(self, text: str, chunk_size: int = 800):
        """Split text into chunks"""
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def preprocess_documents(self):
        """Preprocess documents"""

        # Check cache
        if self.chunks_cache.exists():
            print("\nCache exist, lodaing")
            with open(self.chunks_cache, 'rb') as f:
                self.doc_chunks = pickle.load(f)
            print(f"\nLoad {len(self.doc_chunks)} from cache")
            return True

        for doc in self.documents:
            chunks = self.split_text_chunks(doc["content"])
            for i, chunk in enumerate(chunks):
                self.doc_chunks.append({
                    'content': chunk,
                    'source': doc["filename"],
                    'chunk_id': i,
                    'full_path': doc["path"]
                })
        
        # Write cache
        with open(self.chunks_cache, 'wb') as f:
            pickle.dump(self.doc_chunks, f)
    
    def build_vector_index(self):
        """Build vector index"""

        #. Check cache
        if self.vectors_cache.exists():
            print("\nVector cache exist, loading...")
            with open(self.vectors_cache, 'rb') as f:
                cache_data = pickle.load(f)
                self.vectorizer = TfidfVectorizer(
                    vocabulary = cache_data['vocabulary']
                )
                self.vectorizer.idf_ = cache_data['idf']
                self.doc_vectors = cache_data['vectors']
            print("\nVector index loading complete")
            return True

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,
            min_df = 1,
            max_df = 0.95,
            ngram_range=(1,2)
        )
        text = [chunk["content"] for chunk in self.doc_chunks]
        self.doc_vectors = self.vectorizer.fit_transform(text)
        print("\nTF-IDF index built successfully")

        # Write cache
        cache_data = {
            'vectors': self.doc_vectors,
            'vocabulary': self.vectorizer.vocabulary_,
            'idf': self.vectorizer.idf_
        }

        with open(self.vectors_cache, 'wb') as f:
            pickle.dump(cache_data, f)

        print("\nVector index build complete")

    def search_chunks(self, query: str, top_k:int = 10, similarity_threshold: float = 0.01):
        if self.vectorizer is None or self.doc_vectors is None:
            raise ValueError("Vector index not build, please use build_vector_index()")
        
        # Vectorize querying
        query_vector = self.vectorizer.transform([query])

        # Calculate cos similarity [0,1]
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        print(f"\nSimilarities: {similarities}")

        valid_index = np.where(similarities > similarity_threshold)[0]
        # Arrange similarity ascending
        sorted_index = valid_index[np.argsort(similarities[valid_index])[::-1]]
        # print(sorted_index, "sorted_index")

        top10_index = sorted_index[:top_k]

        results = []
        for idx in top10_index:
            chunk = self.doc_chunks[idx].copy()
            chunk["similarity"] = float(similarities[idx])
            results.append(chunk)
        return results

    def generate_answer(self, question: str, relevant_chunks: list):

        context_parts = []

        for i, chunk in enumerate(relevant_chunks, 1):

            context_parts.append(
                f"""
                [Source {i}]
                File: {chunk['source']}
                Similarity: {chunk['similarity']:.4f}

                Text:
                {chunk['content']}
                """
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
            You are a helpful Sherlock Holmes literary assistant.

            Answer the question ONLY using the context below.

            If the answer is not found, say:
            "I could not find enough information."

            Always cite sources like [Source 1].

            Context:
            {context}

            Question:
            {question}
            """

        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return response.output_text

    def ask(self, question: str):
        # Search related documents
        relevant_chunks = self.search_chunks(question)
        # print(relevant_chunks, 'relevant_chunks')
        if not relevant_chunks:
            return {
                'question': question,
                'answer': 'Sorry, result not found.',
                'sources': []
            }
        # print(f"{len(relevant_chunks)} found")

        # Call LLM API

        answer = self.generate_answer(
            question,
            relevant_chunks
        )

        print("\n=== Answer ===\n")
        print(answer)

        return {
            'question': question,
            'answer': answer,
            'sources': relevant_chunks
        }

def main():
    rag = RagSystem()
    rag.initialize()

    # Interactive chat
    print("\n=== Sherlock Lib System ===")
    print("\nType your question, type 'quit' or 'exit' to exit")

    while True:
        question = input("Please enter your question: ").strip()

        if question.lower() in ['quit', 'exit']:
            print("\nThanks for using")
            break
        if not question:
            continue

        # Get answer
        rag.ask(question)

if __name__ == "__main__":
    main()
