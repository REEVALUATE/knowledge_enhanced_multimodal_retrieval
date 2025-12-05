import requests
from typing import Union, List, Dict
from src.clip.clip_retrieval import CLIPRetrieval
from src.text2sparql.text2sparql_retrieval import TEXT2SPARQLRetrieval

from dotenv import load_dotenv
import os
from io import BytesIO
load_dotenv()

class RetrievalEngine:
    
    def __init__(self):

        self.clip_retriever = CLIPRetrieval()
        self.t2s_retriever = TEXT2SPARQLRetrieval()
        self.cir_endpoint = os.getenv("CIR_ENDPOINT")
        self.cir_headers = {
            'accept': 'application/json',
            'X-API-Key': os.getenv("CIR_ENDPOINT_KEY"),
        }

    def _fuse_clip_sparql_linear(
        self,
        clip_results: List[Dict],
        sparql_results: List[str],
        alpha: float = 0.8,
        beta: float = 0.2
    ) -> List[Dict]:
        """
        Linear fusion of CLIP + SPARQL retrieval results without normalization.
        
        CLIP scores are already normalized (cosine similarity in [-1, 1] or [0, 1]),
        so we don't apply min-max normalization which would artificially boost 
        irrelevant results.

        Args:
            clip_results (List[Dict]): List of dicts with keys {"uuid": str, "score": float}.
                                        Scores are expected to be already normalized.
            sparql_results (List[str]): List of UUIDs returned by SPARQL (no scores).
            alpha (float): Weight for CLIP similarity (default=0.8).
            beta (float): Weight for SPARQL hit bonus (default=0.2).

        Returns:
            List[Dict]: Fused results, sorted by score descending.
                        Each item has: {"uuid", "clip_score", "sparql_hit", "score"}.
        """
        
        if not clip_results:
            return []
        
        # Convert sparql_results to set for O(1) lookup
        sparql_set = set(sparql_results)
        
        fused = []
        for item in clip_results:
            uuid = item["uuid"]
            clip_score = item["score"]  # Use raw CLIP score (already normalized)
            
            # Check if in SPARQL results
            sparql_hit = uuid in sparql_set
            
            # Linear fusion: alpha * CLIP + beta * SPARQL_indicator
            score = alpha * clip_score + beta * (1.0 if sparql_hit else 0.0)
            
            fused.append({
                "uuid": uuid,
                # "clip_score": round(clip_score, 4),
                # "sparql_hit": sparql_hit,
                "score": round(score, 4)
            })
        
        # Sort by score descending
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        return fused



    def retrieve_cir(self, image: bytes, query_text: str, threshold: float = 0.0):
        """
        Call remote CIR API with uploaded image and query text.
        image: bytes (from UploadFile.read())
        query_text: textual query to filter results
        limit: number of results
        threshold: minimum score threshold
        """
        params = {"query": query_text, "limit": 100000}

        files = {
            "file": ("uploaded.jpg", BytesIO(image), "image/jpeg")
        }

        response = requests.post(
            self.cir_endpoint,
            params=params,
            headers=self.cir_headers,
            files=files,       # FIXED here
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"CIR API error {response.status_code}: {response.text}")

        try:
            data = response.json()
        except Exception as e:
            raise Exception(f"Invalid JSON response: {e}")

        print(f"CIR API returned {len(data)} results")
        return [
            {"uuid": item["image_id"], "score": item["score"]}
            for item in data
            if item.get("score", 0) >= threshold
        ]

    def retrieve_text(self, query: str, alpha: float = 0.8, beta: float = 0.2, alpha_clip: float = 0.5, threshold: float = 0):
        clip_results = self.clip_retriever.retrieval(query, alpha=alpha_clip)
        t2s_results = self.t2s_retriever.retrieval(query)
        fused_results =  self._fuse_clip_sparql_linear(
            clip_results=clip_results,
            sparql_results=t2s_results,
            alpha=alpha,
            beta=beta
        )
        return [
            {
                "uuid": item['uuid'],
                "score": item["score"]
            }
            for item in fused_results
            if item.get("score", 0) >= threshold
        ]
    
    def retrieve_text_noknowledge(self, query: str, alpha: float = 0.8, beta: float = 0.2, alpha_clip: float = 0.5, threshold: float = 0):
        fused_results = self.clip_retriever.retrieval(query, alpha=alpha_clip)

        return [
            {
                "uuid": item['uuid'],
                "score": item["score"]
            }
            for item in fused_results
            if item.get("score", 0) >= threshold
        ]