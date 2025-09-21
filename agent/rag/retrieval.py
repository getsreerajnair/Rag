import os, re
from rank_bm25 import BM25Okapi
import numpy as np

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

class DocumentRetriever:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = docs_path
        self.docs = []
        self._build_index()

    def _build_index(self):
        chunk_id = 0
        corpus = []
        for fname in os.listdir(self.docs_path):
            fpath = os.path.join(self.docs_path, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
                # simple split by double newlines or lines starting with
                paragraphs = re.split(r'\n\n', text)
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        self.docs.append({
                            "chunk_id": f"{fname}::chunk{chunk_id}",
                            "content": para,
                            "source": fname
                        })
                        corpus.append(tokenize(para))
                        chunk_id += 1
        print(f"List of chunks created : {self.docs}\n")
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 3):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [ {**self.docs[idx], "score": float(scores[idx])} for idx in top_indices ]
    
    # Add a __call__ method to make the class callable
    def __call__(self, question: str, top_k: int = 3):
        return self.retrieve(question, top_k)


if __name__ == "__main__":
    retriever = DocumentRetriever("docs")
    answer = retriever.retrieve("return policy")
    print(f"result:{answer}")
