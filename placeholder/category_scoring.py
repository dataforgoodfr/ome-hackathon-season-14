# scorer.py
import numpy as np
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer, util
import json


class ThemeSimilarityScorer:
    """
    Scoring multi-thèmes via embeddings, pour textes segmentés (~100 mots).
    - Centroïde = moyenne des embeddings de prototypes du thème
    - Score = cosine similarity avec centroïde (dot car embeddings normalisés)
    - Sortie = vecteur de similarités (un score par thème)
    """

    def __init__(
        self,
        themes_prototypes: Dict[str, List[str]],
        theme_order: List[str],
        model_name: str = "intfloat/multilingual-e5-base",
        use_e5_prefix: bool = True,
    ):
        self.themes_prototypes = themes_prototypes
        self.theme_order = theme_order
        self.model_name = model_name
        self.use_e5_prefix = use_e5_prefix

        self.model = SentenceTransformer(model_name)

        # precompute prototypes embeddings + centroids
        self.proto_embs = self._precompute_prototypes()
        self.centroids = self._build_centroids()

    def _prefix(self, text: str, is_query: bool) -> str:
        if not self.use_e5_prefix:
            return text
        return ("query: " + text) if is_query else ("passage: " + text)

    def _embed(self, text: str, is_query: bool) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            # vecteur zéro (dimension inconnue à ce stade) => on gère au niveau score
            return None
        text = self._prefix(text, is_query=is_query)
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

    def _precompute_prototypes(self) -> Dict[str, np.ndarray]:
        proto_embs = {}
        for theme, protos in self.themes_prototypes.items():
            protos_in = [self._prefix(p.strip(), is_query=False) for p in protos]
            E = self.model.encode(protos_in, normalize_embeddings=True, convert_to_numpy=True)
            proto_embs[theme] = E  # (K, d)
        return proto_embs

    def _build_centroids(self) -> Dict[str, np.ndarray]:
        centroids = {}
        for theme in self.theme_order:
            E = self.proto_embs[theme]
            c = E.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            centroids[theme] = c.astype(np.float32)
        return centroids

    def score(self, text: str, return_details: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Score un texte (segment). Retourne:
          - scores_vector: np.array([sim_t1, sim_t2, ...])
          - details: dict
        """
        emb = self._embed(text, is_query=True)

        if emb is None:
            scores = np.zeros(len(self.theme_order), dtype=float)
            details = {
                "scores_by_theme": dict(zip(self.theme_order, scores.tolist())),
                "best_theme": None,
                "best_score": 0.0,
                "empty_text": True,
            }
            return scores, details

        scores = np.array([float(np.dot(emb, self.centroids[t])) for t in self.theme_order], dtype=float)

        details = {
            "scores_by_theme": dict(zip(self.theme_order, scores.tolist())),
            "best_theme": self.theme_order[int(np.argmax(scores))],
            "best_score": float(np.max(scores)),
            "empty_text": False,
        }

        if return_details:
            closest = {}
            for theme in self.theme_order:
                sims = util.cos_sim(emb, self.proto_embs[theme])[0].cpu().numpy()
                idx = int(np.argmax(sims))
                closest[theme] = {
                    "prototype": self.themes_prototypes[theme][idx],
                    "prototype_score": float(sims[idx]),
                }
            details["closest_prototypes"] = closest

        return scores, details

    def multilabel(self, scores_vector: np.ndarray, thresholds: Dict[str, float]) -> List[str]:
        labels = []
        for i, theme in enumerate(self.theme_order):
            if scores_vector[i] >= thresholds.get(theme, 0.0):
                labels.append(theme)
        return labels


def default_scorer() -> ThemeSimilarityScorer:
    """
    Factory: construit un scorer par défaut (à personnaliser).
    """
    theme_order = ["agriculture_alimentation", "mobility_transport", "energy"]




    with open("placeholder/files/themes_prototypes_200.json", "r", encoding="utf-8") as f:
        themes_prototypes = json.load(f)



    return ThemeSimilarityScorer(
        themes_prototypes=themes_prototypes,
        theme_order=theme_order,
        model_name="intfloat/multilingual-e5-base",
        use_e5_prefix=True,
    )
