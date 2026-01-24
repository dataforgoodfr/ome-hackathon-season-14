
"""
Embedding-based multi-theme scoring (3 themes) for LONG documents:
- Build 1 centroid per theme from prototypes (mean embedding)
- Chunk long documents
- Compute cosine similarity of each chunk to each centroid
- Aggregate per-theme by taking mean of TOP-K chunk scores (evidence-based)
- Output = vector of similarities [agri, mobility, energy]
- Optional: multi-label with per-theme thresholds
- Optional: return top evidence chunks per theme (explainability)

Dependencies:
  pip install -U sentence-transformers numpy

Recommended embedding model (good for French):
  intfloat/multilingual-e5-base
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "intfloat/multilingual-e5-base"
USE_E5_PREFIX = True  # E5 performs better with "query:" / "passage:" prefixes

THEME_ORDER = ["agriculture_alimentation", "mobility_transport", "energy"]

# Chunking for long docs
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40

# Evidence aggregation
TOPK_CHUNKS_PER_THEME = 3
AGG_METHOD = "mean_topk"  # "mean_topk" or "max"

# Optional thresholds for multi-label outputs (tune with your data)
DEFAULT_THRESHOLDS = {
    "agriculture_alimentation": 0.35,
    "mobility_transport": 0.35,
    "energy": 0.35,
}

# Prototypes (generic baseline — strongly recommended to enrich with your domain vocabulary)
THEMES_PROTOTYPES: Dict[str, List[str]] = {
    "agriculture_alimentation": [
        "Production agricole et exploitation des terres.",
        "Transformation et distribution des produits alimentaires.",
        "Systèmes alimentaires durables et circuits courts.",
        "Gestion de l’élevage et pratiques agro-écologiques.",
        "Sécurité alimentaire, qualité sanitaire et nutrition.",
        "Agriculture de précision, capteurs et innovations agrotech.",
        "Politiques agricoles, subventions et réglementation agroalimentaire.",
        "Gestion de l’eau et des sols pour la culture et l’irrigation.",
        "Chaîne de valeur agroalimentaire du producteur au consommateur.",
        "Impacts environnementaux des pratiques agricoles et de l’alimentation.",
        "Approvisionnement, stockage et logistique des denrées alimentaires.",
        "Étiquetage, traçabilité et contrôle qualité des aliments."
    ],
    "mobility_transport": [
        "Modes de déplacement et infrastructures de transport.",
        "Mobilité urbaine, transports publics et intermodalité.",
        "Mobilité douce : vélo, marche, micro-mobilité.",
        "Logistique et transport de marchandises, fret et supply chain.",
        "Réseaux routier, ferroviaire, aérien et maritime.",
        "Réduction des émissions liées au transport et décarbonation.",
        "Véhicules électriques, hybrides et bornes de recharge.",
        "Gestion du trafic, planification des déplacements et congestion.",
        "Smart mobility, MaaS et services numériques de mobilité.",
        "Sécurité routière et réglementation des transports.",
        "Optimisation des flux, itinéraires et livraison du dernier kilomètre.",
        "Infrastructures pour la mobilité durable et aménagement urbain."
    ],
    "energy": [
        "Production, distribution et stockage d’énergie.",
        "Énergies renouvelables : solaire, éolien, hydraulique, biomasse.",
        "Réseaux électriques, smart grids et flexibilité énergétique.",
        "Efficacité énergétique des bâtiments et de l’industrie.",
        "Transition énergétique et réduction des émissions de CO2.",
        "Nucléaire, thermique et mix énergétique.",
        "Marchés de l’énergie, prix, régulation et fournisseurs.",
        "Consommation énergétique, sobriété et pilotage de la demande.",
        "Autoconsommation, production locale et micro-réseaux.",
        "Stockage : batteries, hydrogène, STEP et solutions long terme.",
        "Réseaux gaz, chaleur, électricité et infrastructures énergétiques.",
        "Innovation : hydrogène, capture carbone, nouvelles technologies énergétiques."
    ],
}


# -----------------------------
# TEXT UTILS
# -----------------------------
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def chunk_text_words(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple word-based chunking; replace with sentence/section-aware chunking if you can."""
    text = normalize_whitespace(text)
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    step = max(1, chunk_words - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_words])
        if chunk:
            chunks.append(chunk)
        if i + chunk_words >= len(words):
            break
    return chunks

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)


# -----------------------------
# CORE: CENTROIDS + SCORING
# -----------------------------

import numpy as np
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer, util


class ThemeSimilarityScorer:
    """
    Scorer embeddings pour textes SEGMENTÉS (~100 mots) :
    - Pas de chunking
    - 1 embedding par segment
    - 1 centroïde par thème construit à partir de prototypes
    - Output : vecteur de similarités par thème + explications optionnelles
    """

    def __init__(
        self,
        themes_prototypes: Dict[str, List[str]],
        theme_order: List[str],
        model_name: str = "intfloat/multilingual-e5-base",
        use_e5_prefix: bool = True,
        precompute_proto_embeddings: bool = True,
    ):
        """
        themes_prototypes: {theme: [proto1, proto2, ...]}
        theme_order: ordre des thèmes dans le vecteur de sortie
        """
        self.themes_prototypes = themes_prototypes
        self.theme_order = theme_order
        self.use_e5_prefix = use_e5_prefix
        self.model = SentenceTransformer(model_name)

        # Encodage des prototypes (optionnel, mais recommandé)
        self.proto_embs = {}
        if precompute_proto_embeddings:
            self._precompute_prototypes()

        # Centroïdes
        self.centroids = self._build_centroids()

    def _prefix(self, text: str, is_query: bool) -> str:
        if not self.use_e5_prefix:
            return text
        return ("query: " + text) if is_query else ("passage: " + text)

    def _embed(self, text: str, is_query: bool) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            # vecteur nul si texte vide
            return np.zeros(768, dtype=np.float32)
        text = self._prefix(text, is_query=is_query)
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

    def _precompute_prototypes(self) -> None:
        for theme, protos in self.themes_prototypes.items():
            protos_in = [self._prefix(p.strip(), is_query=False) for p in protos]
            embs = self.model.encode(
                protos_in,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            self.proto_embs[theme] = embs  # shape (K, d)

    def _build_centroids(self) -> Dict[str, np.ndarray]:
        centroids = {}
        for theme in self.theme_order:
            protos = self.themes_prototypes[theme]

            # Si embeddings prototypes déjà calculés, utilise-les
            if theme in self.proto_embs:
                E = self.proto_embs[theme]
            else:
                protos_in = [self._prefix(p.strip(), is_query=False) for p in protos]
                E = self.model.encode(
                    protos_in,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )

            c = E.mean(axis=0)
            # re-normalize
            c = c / (np.linalg.norm(c) + 1e-12)
            centroids[theme] = c.astype(np.float32)

        return centroids

    def score_segment(
        self,
        text: str,
        return_details: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Score un segment (~100 mots).

        Returns:
          scores_vector: np.array([sim_theme1, sim_theme2, sim_theme3]) selon theme_order
          details: infos optionnelles (scores par thème, prototype le plus proche, etc.)
        """
        emb = self._embed(text, is_query=True)

        # Similarité à chaque centroïde (cosine = dot car normalisés)


# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    scorer = ThemeSimilarityScorer(
        themes_prototypes=THEMES_PROTOTYPES,
        model_name=MODEL_NAME,
        use_e5_prefix=USE_E5_PREFIX,
        theme_order=THEME_ORDER
    )

    # Example long-ish text (replace by your docs)
    doc = """
    Le projet décrit le déploiement de bornes de recharge et l’électrification de flottes,
    ainsi que l’optimisation des flux logistiques et du dernier kilomètre. Une partie
    présente aussi des aspects de flexibilité du réseau, stockage batterie, et intégration
    d’énergies renouvelables pour sécuriser l’approvisionnement énergétique.
    """ * 10

    scores, meta = scorer.score_long_document(
        doc,
        chunk_words=CHUNK_WORDS,
        overlap=CHUNK_OVERLAP,
        topk=TOPK_CHUNKS_PER_THEME,
        agg=AGG_METHOD,
        return_evidence=True
    )

    # Output vector of similarities (in THEME_ORDER)
    result = dict(zip(THEME_ORDER, [round(x, 3) for x in scores]))
    print("Similarity vector:", result)

    # Optional multi-label
    labels = scorer.multilabel_from_scores(scores, thresholds=DEFAULT_THRESHOLDS)
    print("Multi-label (thresholded):", labels)

    # Optional evidence (top chunks) for explainability
    # print(meta["evidence"])  # can be verbose
    for theme in THEME_ORDER:
        print(f"\nTop evidence for {theme}:")
        for ev in meta["evidence"][theme]:
            print("  -", ev["score"], ":", ev["chunk"])
