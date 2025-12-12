import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Union, Any, Iterator

# Try to import Numba optimizations (optional)
try:
    from numba_optimizations import (
        dcaap_activation_jit,
        gelu_jit,
        sigmoid_jit,
        matmul_add_jit,
        softmax_jit,
        distance_squared_jit,
        NUMBA_AVAILABLE,
    )
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    NUMBA_AVAILABLE = False

"""
Implementation of T-RLINKOS TRM++ without external deep learning frameworks.

This version removes the dependency on PyTorch (which raised
`ModuleNotFoundError: No module named 'torch'`) and instead uses pure NumPy.

The goal is to keep the **logic and architecture** of the original design:
- dCaAP-inspired neuron cell (DCaAPCell)
- Torque Clustering router (TorqueRouter)
- TRM core (TRLinkosCore)
- Fractal Merkle-DAG for reasoning trace
- TRLinkosTRM high-level recursive loop

So you can run and experiment with the model even in environments
where `torch` is not available.

Optional Numba/JIT Optimization:
- If numba is installed, performance-critical operations are JIT-compiled
- Provides 2-5x speedup for large batches without code changes
- Falls back gracefully to pure NumPy if numba is unavailable
"""

# ============================
#  Helper layers and activations (NumPy-based)
# ============================


class LinearNP:
    """Simple fully-connected layer using NumPy.

    y = x @ W.T + b

    - x: [B, in_features]
    - W: [out_features, in_features]
    - b: [out_features]
    - output: [B, out_features]
    """

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        # He-like initialization
        limit = np.sqrt(2.0 / max(1, in_features))
        self.W = np.random.uniform(-limit, limit, (out_features, in_features))
        self.b = np.zeros(out_features, dtype=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if USE_NUMBA and NUMBA_AVAILABLE:
            return matmul_add_jit(x, self.W, self.b)
        return x @ self.W.T + self.b


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (approximation).
    
    Uses JIT-compiled version if Numba is available for ~2-3x speedup.
    """
    if USE_NUMBA and NUMBA_AVAILABLE:
        return gelu_jit(x)
    # Approximation via tanh (Hendrycks & Gimpel)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax implementation.
    
    Uses JIT-compiled version if Numba is available for ~2x speedup.
    Note: Converts to float64 for numerical stability with large values.
    """
    if USE_NUMBA and NUMBA_AVAILABLE and axis == -1:
        return softmax_jit(x, axis)
    # Ensure float type for numerical stability (handles integer inputs)
    x = np.asarray(x, dtype=np.float64 if x.dtype.kind in ('i', 'u') else None)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================
#  dCaAP-inspired neuron cell
# ============================

def dcaap_activation(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Activation dCaAP (dendritic Calcium Action Potential).

    Basée sur Gidon et al., Science 2020 et Hashemi & Tetzlaff, bioRxiv 2025.
    Référence: https://www.biorxiv.org/content/10.1101/2025.06.10.658823v1

    La fonction dCaAP est non-monotone et permet la détection d'anti-coïncidence:
    - Amplitude maximale proche du seuil
    - Amplitude réduite pour des stimuli très forts
    - Zéro pour les stimuli sous le seuil

    dCaAP(x) = 4 * σ(x-θ) * (1 - σ(x-θ)) * (x > θ)

    Cette activation permet à un seul neurone de résoudre le problème XOR,
    ce que ReLU ne peut pas faire.
    
    Uses JIT-compiled version if Numba is available for ~3-5x speedup.

    Args:
        x: Entrée [B, D]
        threshold: Seuil d'activation θ

    Returns:
        Activation dCaAP [B, D]
    """
    if USE_NUMBA and NUMBA_AVAILABLE:
        return dcaap_activation_jit(x, threshold)
    x_shifted = x - threshold
    sigmoid_x = 1.0 / (1.0 + np.exp(-x_shifted))
    # Produit de sigmoïde et son complément = forme en cloche
    dcaap = 4.0 * sigmoid_x * (1.0 - sigmoid_x)
    # Masque pour les valeurs au-dessus du seuil
    mask = (x > threshold).astype(np.float64)
    return dcaap * mask


class DCaAPCell:
    """Neurone inspiré dCaAP (Gidon et al., Science 2020; Hashemi & Tetzlaff, bioRxiv 2025).

    Implémente le modèle de potentiel d'action calcique dendritique:
    - Branches dendritiques multiples avec intégration locale
    - Activation dCaAP non-monotone (détection d'anti-coïncidence)
    - Gate calcique pour l'accumulation temporelle
    - Capacité XOR intrinsèque (contrairement aux activations standard)

    Référence: https://www.biorxiv.org/content/10.1101/2025.06.10.658823v1

    Architecture:
    - Entrée : concat(x, y, z) distribué sur branches dendritiques
    - Branches dendritiques : intégration locale avec activation dCaAP
    - Calcium gate : accumulation + seuil adaptatif
    - Sortie : nouveau z (état interne)
    """

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, num_branches: int = 4):
        self.num_branches = num_branches
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        # Dimension de sortie par branche dendritique
        self.branch_dim = hidden_dim // num_branches

        # Synapses pour chaque branche dendritique
        self.branch_weights = [
            LinearNP(input_dim, self.branch_dim) for _ in range(num_branches)
        ]

        # Seuils adaptatifs pour chaque branche (paramètres appris)
        limit = np.sqrt(2.0 / max(1, self.branch_dim))
        self.branch_thresholds = [
            np.random.uniform(-limit, limit, (1, self.branch_dim)) for _ in range(num_branches)
        ]

        # Intégration somatique des branches
        self.soma_integration = LinearNP(hidden_dim, hidden_dim)

        # Gate calcique pour l'accumulation temporelle
        self.calcium_gate = LinearNP(hidden_dim, 1)

        # Projection vers l'espace z
        self.output_projection = LinearNP(hidden_dim, z_dim)

    def forward(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Propagation avant avec mécanisme dCaAP.

        Implémente l'intégration dendritique avec activation dCaAP:
        1. Distribution de l'entrée sur les branches dendritiques
        2. Intégration locale avec activation dCaAP non-monotone
        3. Sommation somatique des signaux dendritiques
        4. Gate calcique pour la mise à jour de l'état

        Args:
            x: [B, dx] - Entrée externe
            y: [B, dy] - Réponse courante
            z: [B, dz] - État interne

        Returns:
            z_next: [B, dz] - Nouvel état interne
        """
        h_in = np.concatenate([x, y, z], axis=-1)  # [B, input_dim]

        # 1) Intégration par branches dendritiques avec activation dCaAP
        branch_outputs = []
        for i in range(self.num_branches):
            # Projection synaptique vers la branche
            branch_input = self.branch_weights[i](h_in)  # [B, branch_dim]

            # Activation dCaAP avec seuil adaptatif
            # Le seuil peut varier par branche (hétérogénéité dendritique)
            threshold = np.mean(self.branch_thresholds[i])
            branch_activation = dcaap_activation(branch_input, threshold)

            branch_outputs.append(branch_activation)

        # 2) Concatenation des sorties des branches
        dendritic_signal = np.concatenate(branch_outputs, axis=-1)  # [B, hidden_dim]

        # 3) Intégration somatique
        soma_potential = self.soma_integration(dendritic_signal)  # [B, hidden_dim]
        soma_potential = gelu(soma_potential)  # Non-linéarité somatique

        # 4) Gate calcique pour l'accumulation temporelle
        ca_potential = self.calcium_gate(soma_potential)  # [B, 1]
        # Sigmoid pour gate entre 0 et 1
        gate = 1.0 / (1.0 + np.exp(-ca_potential))

        # 5) Projection vers l'espace z et mise à jour
        proposal = self.output_projection(soma_potential)  # [B, z_dim]

        # Mise à jour gated de l'état (interpolation contrôlée par calcium)
        z_next = z + gate * (proposal - z)

        return z_next


# ============================
#  Torque Clustering Router
# ============================


class TorqueRouter:
    """Routeur basé sur Torque Clustering (Yang & Lin, TPAMI 2025).

    Implémente le concept de Torque Clustering pour le routage d'experts:
    - Torque = Mass × R² (distance squared)
    - Mass: densité locale basée sur les représentations
    - R²: distance au carré vers les centroïdes d'experts

    Référence: https://github.com/JieYangBruce/TorqueClustering
    """

    def __init__(self, x_dim: int, y_dim: int, z_dim: int, num_experts: int):
        self.num_experts = num_experts
        self.input_dim = x_dim + y_dim + z_dim

        # Projection pour calculer les représentations
        self.projection = LinearNP(self.input_dim, 64)

        # Centroïdes des experts (appris)
        limit = np.sqrt(2.0 / 64)
        self.expert_centroids = np.random.uniform(-limit, limit, (num_experts, 64))

        # Paramètres pour le calcul de masse locale
        self.mass_projection = LinearNP(self.input_dim, 1)

    def _compute_distance_matrix(self, h: np.ndarray) -> np.ndarray:
        """Calcule la matrice de distances carrées entre h et les centroïdes.

        h: [B, D] - représentations projetées
        returns: [B, E] - distances carrées vers chaque centroïde d'expert
        
        Uses JIT-compiled version if Numba is available for ~3-4x speedup.
        """
        # Use optimized version if available
        if USE_NUMBA and NUMBA_AVAILABLE:
            return distance_squared_jit(h, self.expert_centroids)
        
        # h: [B, D], centroids: [E, D]
        # distance² = ||h - c||² = ||h||² + ||c||² - 2*h@c.T
        h_sq = np.sum(h ** 2, axis=1, keepdims=True)  # [B, 1]
        c_sq = np.sum(self.expert_centroids ** 2, axis=1, keepdims=True).T  # [1, E]
        cross = h @ self.expert_centroids.T  # [B, E]
        dist_sq = h_sq + c_sq - 2 * cross  # [B, E]
        return np.maximum(dist_sq, 0.0)  # Assurer non-négatif

    def _compute_mass(self, h_raw: np.ndarray) -> np.ndarray:
        """Calcule la masse locale (densité) pour chaque échantillon.

        Inspiré du concept de masse dans Torque Clustering:
        mass = product of community sizes in the original algorithm.
        Ici, on l'approxime par une mesure de "densité de représentation".

        h_raw: [B, input_dim]
        returns: [B, 1]
        """
        # Utiliser une projection apprise pour estimer la masse
        mass_logit = self.mass_projection(h_raw)  # [B, 1]
        # Masse positive via softplus
        mass = np.log1p(np.exp(mass_logit)) + 1.0  # [B, 1], minimum 1.0
        return mass

    def forward(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Calcule les poids de routage basés sur Torque Clustering.

        Torque = Mass × R² pour chaque paire (échantillon, expert)
        Les experts avec un torque élevé reçoivent plus de poids.

        x: [B, dx]
        y: [B, dy]
        z: [B, dz]
        returns: [B, E] - poids de routage normalisés
        """
        h_raw = np.concatenate([x, y, z], axis=-1)  # [B, input_dim]

        # 1) Projection dans l'espace de représentation
        h = self.projection(h_raw)  # [B, 64]
        h = gelu(h)

        # 2) Calcul des distances carrées (R²) vers les centroïdes d'experts
        R_squared = self._compute_distance_matrix(h)  # [B, E]

        # 3) Calcul de la masse locale
        mass = self._compute_mass(h_raw)  # [B, 1]

        # 4) Calcul du score de routage basé sur Torque Clustering
        # Note: Dans l'algorithme original, Torque = mass × R² identifie les centres.
        # Pour le routage d'experts, nous adaptons le concept:
        # - Chaque expert est un "centre" potentiel
        # - On calcule l'affinité vers chaque expert
        # - Score élevé = expert approprié pour cette entrée
        # L'affinité utilise l'inverse de R² pondéré par la masse:
        # score = mass / (R² + ε), où les experts proches ont un score plus élevé.
        epsilon = 1e-6
        affinity_score = mass / (R_squared + epsilon)  # [B, E]

        # 5) Normalisation via softmax pour obtenir les poids
        weights = softmax(affinity_score, axis=-1)  # [B, E]

        return weights

    def forward_sparse(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        top_k: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Routage sparse vers les top-k experts seulement.

        Implémente un routage sparse qui active uniquement les top-k experts
        les plus pertinents pour chaque échantillon. Cela permet:
        - Réduction de 50-75% du calcul (selon k)
        - Spécialisation accrue des experts
        - Meilleure interprétabilité du routage

        Args:
            x: Entrée externe [B, dx]
            y: Réponse courante [B, dy]
            z: État interne [B, dz]
            top_k: Nombre d'experts à activer (1 à num_experts-1)

        Returns:
            sparse_weights: Poids normalisés [B, E] avec (E - top_k) zéros
            top_indices: Indices des experts sélectionnés [B, top_k]
        """
        # Obtenir les poids complets
        weights = self.forward(x, y, z)  # [B, E]

        # Valider top_k
        top_k = max(1, min(top_k, self.num_experts))

        # Sélectionner top-k experts pour chaque échantillon
        top_indices = np.argsort(weights, axis=-1)[:, -top_k:]  # [B, top_k]

        # Créer les poids sparse
        sparse_weights = np.zeros_like(weights)  # [B, E]

        for i in range(weights.shape[0]):
            sparse_weights[i, top_indices[i]] = weights[i, top_indices[i]]

        # Re-normaliser pour que la somme soit 1
        row_sums = sparse_weights.sum(axis=-1, keepdims=True)
        sparse_weights = sparse_weights / (row_sums + 1e-10)

        return sparse_weights, top_indices


# ============================
#  DivergenceDetector - Détection de divergence du raisonnement
# ============================

# Numerical tolerance constant for avoiding division by zero
NUMERICAL_TOLERANCE = 1e-10


class DivergenceDetector:
    """Détecte la divergence du raisonnement en temps réel.

    Cette classe surveille la qualité du raisonnement et détecte quand
    le processus diverge (hallucinations, erreurs de raisonnement).

    Utilise plusieurs heuristiques:
    1. Variance des scores sur fenêtre glissante
    2. Distance cosinus entre états consécutifs
    3. Gradient du score (positif = convergence)

    Example:
        >>> detector = DivergenceDetector()
        >>> for step in range(max_steps):
        ...     score, state = reasoning_step(...)
        ...     detector.update(score, state)
        ...     is_div, reason = detector.is_diverging()
        ...     if is_div:
        ...         print(f"Divergence detected: {reason}")
        ...         # Trigger backtracking
    """

    def __init__(
        self,
        window_size: int = 5,
        variance_threshold: float = 0.1,
        cosine_threshold: float = 0.95,
        gradient_threshold: float = -0.01
    ):
        """Initialise le détecteur de divergence.

        Args:
            window_size: Taille de la fenêtre glissante pour l'historique
            variance_threshold: Seuil de variance des scores au-delà duquel
                                on considère une divergence
            cosine_threshold: Seuil de similarité cosinus en dessous duquel
                              on détecte une discontinuité d'état
            gradient_threshold: Seuil de gradient du score en dessous duquel
                                on détecte une dégradation
        """
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.cosine_threshold = cosine_threshold
        self.gradient_threshold = gradient_threshold
        self.score_history: List[float] = []
        self.state_history: List[np.ndarray] = []

    def update(self, score: float, state: np.ndarray) -> None:
        """Met à jour l'historique avec une nouvelle observation.

        Args:
            score: Score de la réponse à cette étape
            state: État (y ou z) à cette étape
        """
        self.score_history.append(score)
        self.state_history.append(state.copy())

        # Garder seulement la fenêtre récente
        if len(self.score_history) > self.window_size:
            self.score_history.pop(0)
            self.state_history.pop(0)

    def is_diverging(self) -> Tuple[bool, str]:
        """Détecte si le raisonnement diverge.

        Analyse l'historique récent pour détecter des signes de divergence:
        - Variance élevée des scores → instabilité
        - Faible similarité entre états → discontinuité
        - Gradient négatif des scores → dégradation

        Returns:
            Tuple (is_diverging, reason):
                - is_diverging: True si divergence détectée
                - reason: Description de la raison de la divergence
        """
        if len(self.score_history) < 3:
            return False, "Not enough data"

        # 1. Vérifier la variance des scores
        variance = float(np.var(self.score_history))
        if variance > self.variance_threshold:
            return True, f"High score variance: {variance:.4f}"

        # 2. Vérifier la similarité des états (distance cosinus)
        if len(self.state_history) >= 2:
            last_state = self.state_history[-1].flatten()
            prev_state = self.state_history[-2].flatten()
            norm_last = np.linalg.norm(last_state)
            norm_prev = np.linalg.norm(prev_state)

            if norm_last > NUMERICAL_TOLERANCE and norm_prev > NUMERICAL_TOLERANCE:
                cosine = float(np.dot(last_state, prev_state) / (norm_last * norm_prev))
                if cosine < self.cosine_threshold:
                    return True, f"State discontinuity: cosine={cosine:.4f}"

        # 3. Vérifier le gradient du score
        if len(self.score_history) >= 3:
            gradient = float(np.mean(np.diff(self.score_history[-3:])))
            if gradient < self.gradient_threshold:
                return True, f"Score degradation: gradient={gradient:.4f}"

        return False, "Converging normally"

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'historique.

        Returns:
            Dict avec les statistiques clés:
                - num_observations: Nombre d'observations
                - score_variance: Variance des scores
                - score_mean: Moyenne des scores
                - score_trend: Tendance (gradient moyen)
        """
        if len(self.score_history) == 0:
            return {
                "num_observations": 0,
                "score_variance": 0.0,
                "score_mean": 0.0,
                "score_trend": 0.0,
            }

        return {
            "num_observations": len(self.score_history),
            "score_variance": float(np.var(self.score_history)),
            "score_mean": float(np.mean(self.score_history)),
            "score_trend": float(np.mean(np.diff(self.score_history))) if len(self.score_history) > 1 else 0.0,
        }

    def reset(self) -> None:
        """Réinitialise le détecteur."""
        self.score_history = []
        self.state_history = []


# ============================
#  Coeur de raisonnement TRM : TRLinkosCore
# ============================

class TRLinkosCore:
    """Coeur du Tiny Recursive Model T-RLINKOS (NumPy version).

    - plusieurs experts dCaAP pilotés par TorqueRouter
    - un module pour mettre à jour la réponse y
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden_dim: int = 256,
        num_experts: int = 4,
    ) -> None:
        self.z_dim = z_dim
        self.num_experts = num_experts

        # dCaAP experts
        self.experts = [
            DCaAPCell(x_dim + y_dim + z_dim, hidden_dim, z_dim)
            for _ in range(num_experts)
        ]

        self.router = TorqueRouter(x_dim, y_dim, z_dim, num_experts)

        # update de la réponse y à partir de z
        self.answer_dense1 = LinearNP(y_dim + z_dim, hidden_dim)
        self.answer_dense2 = LinearNP(hidden_dim, y_dim)

    def step_reasoning(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        inner_recursions: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Une "récursion" TRM = plusieurs updates internes de z, puis mise à jour de y.

        Args:
            x: Entrée encodée [B, dx]
            y: État de réponse actuel [B, dy]
            z: État interne actuel [B, dz]
            inner_recursions: Nombre de récursions internes

        Returns:
            y_next: Nouvelle réponse [B, dy]
            z: Nouvel état interne [B, dz]
        """
        # 1) inner recursion sur z
        for _ in range(inner_recursions):
            weights = self.router.forward(x, y, z)  # [B, E]

            # Appliquer tous les experts et empiler les résultats
            # Note: Utilisation de np.stack pour une meilleure efficacité mémoire
            z_stack = np.stack(
                [expert.forward(x, y, z) for expert in self.experts],
                axis=1
            )  # [B, E, dz]

            # Mélange pondéré par les poids Torque
            weights_expanded = weights[:, :, None]  # [B, E, 1]
            z = np.sum(z_stack * weights_expanded, axis=1)  # [B, dz]

        # 2) mise à jour de la réponse
        y_in = np.concatenate([y, z], axis=-1)
        h = self.answer_dense1(y_in)
        h = gelu(h)
        y_next = self.answer_dense2(h)
        return y_next, z


# ============================
#  Merkle-DAG fractal pour tracer le raisonnement
# ============================


def hash_tensor(t: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()


@dataclass
class DAGNode:
    """Noeud du Merkle-DAG fractal représentant un état de raisonnement.

    Structure fractale: chaque noeud peut contenir un sous-DAG (branche),
    créant une auto-similarité à plusieurs échelles.

    Attributes:
        node_id: Identifiant unique SHA256 du noeud
        step: Étape de raisonnement (0 = début)
        depth: Profondeur fractale (0 = niveau racine)
        y_hash: Hash SHA256 de l'état y
        z_hash: Hash SHA256 de l'état z
        parents: Liste des node_ids parents (historique)
        children: Liste des node_ids enfants (branches)
        score: Score optionnel de la réponse
        y_state: État y stocké pour le backtracking
        z_state: État z stocké pour le backtracking
        branch_root: ID du noeud racine de la branche fractale (si applicable)
    """
    node_id: str
    step: int
    y_hash: str
    z_hash: str
    depth: int = 0
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    score: Optional[float] = None
    y_state: Optional[np.ndarray] = field(default=None, repr=False)
    z_state: Optional[np.ndarray] = field(default=None, repr=False)
    branch_root: Optional[str] = None


class FractalMerkleDAG:
    """Merkle-DAG fractal pour T-RLINKOS.

    Structure de données fractale pour tracer et auditer le raisonnement:
    - Chaque étape de raisonnement = un noeud
    - Structure auto-similaire: chaque noeud peut avoir des branches fractales
    - Parents = historique de raisonnement (backtracking possible)
    - Children = branches fractales explorant des alternatives
    - best_node = meilleur score vu jusqu'à présent
    - Stockage optionnel des états y/z pour le backtracking

    La propriété fractale est implémentée par:
    1. Branches multiples à chaque noeud (exploration parallèle)
    2. Profondeur (depth) indiquant le niveau fractal
    3. Sous-DAGs récursifs via branch_root
    """

    def __init__(self, store_states: bool = False, max_depth: int = 3) -> None:
        """Initialise le DAG fractal.

        Args:
            store_states: Si True, stocke les états y/z pour le backtracking
            max_depth: Profondeur fractale maximale (auto-similarité)
        """
        self.nodes: Dict[str, DAGNode] = {}
        self.best_node_id: Optional[str] = None
        self.best_score: float = float("-inf")
        self.store_states = store_states
        self.max_depth = max_depth
        self.root_nodes: List[str] = []  # Noeuds racines par échantillon

    def add_step(
        self,
        step: int,
        y: np.ndarray,
        z: np.ndarray,
        parents: List[str],
        score: Optional[float] = None,
        depth: int = 0,
        branch_root: Optional[str] = None,
    ) -> str:
        """Ajoute une étape de raisonnement au DAG fractal.

        Args:
            step: Numéro de l'étape
            y: État de la réponse [1, dy]
            z: État interne [1, dz]
            parents: Liste des node_ids parents
            score: Score optionnel de la réponse
            depth: Profondeur fractale du noeud
            branch_root: ID du noeud racine de la branche (pour sous-DAG)

        Returns:
            node_id: Identifiant unique du noeud créé
        """
        y_h = hash_tensor(y)
        z_h = hash_tensor(z)
        # Utiliser des séparateurs pour éviter les collisions de hash
        raw = f"{step}|{depth}|{y_h}|{z_h}|{'|'.join(parents)}".encode("utf-8")
        node_id = hashlib.sha256(raw).hexdigest()

        node = DAGNode(
            node_id=node_id,
            step=step,
            y_hash=y_h,
            z_hash=z_h,
            depth=depth,
            parents=list(parents),
            children=[],
            score=score,
            y_state=y.copy() if self.store_states else None,
            z_state=z.copy() if self.store_states else None,
            branch_root=branch_root,
        )
        self.nodes[node_id] = node

        # Mettre à jour les liens parent -> enfant
        for parent_id in parents:
            if parent_id in self.nodes:
                if node_id not in self.nodes[parent_id].children:
                    self.nodes[parent_id].children.append(node_id)

        # Suivre les noeuds racines (noeuds sans parents au niveau principal)
        if not parents and depth == 0:
            self.root_nodes.append(node_id)

        if score is not None and score > self.best_score:
            self.best_score = score
            self.best_node_id = node_id

        return node_id

    def create_branch(
        self,
        parent_node_id: str,
        y: np.ndarray,
        z: np.ndarray,
        score: Optional[float] = None,
    ) -> Optional[str]:
        """Crée une branche fractale à partir d'un noeud existant.

        Implémente l'auto-similarité: une branche est un sous-DAG
        qui explore une alternative à partir du noeud parent.

        Args:
            parent_node_id: ID du noeud parent où créer la branche
            y: État y initial de la branche
            z: État z initial de la branche
            score: Score optionnel

        Returns:
            node_id de la racine de la branche, ou None si profondeur max atteinte
        """
        parent = self.nodes.get(parent_node_id)
        if parent is None:
            return None

        new_depth = parent.depth + 1
        if new_depth > self.max_depth:
            return None  # Limite de profondeur fractale atteinte

        # Créer le noeud racine de la branche
        branch_node_id = self.add_step(
            step=0,  # Nouvelle branche commence à step 0
            y=y,
            z=z,
            parents=[parent_node_id],
            score=score,
            depth=new_depth,
            branch_root=parent_node_id,
        )

        return branch_node_id

    def get_branch_nodes(self, branch_root_id: str) -> List[DAGNode]:
        """Récupère tous les noeuds d'une branche fractale.

        Args:
            branch_root_id: ID du noeud racine de la branche

        Returns:
            Liste des noeuds appartenant à cette branche (excluant la racine)
        """
        return [
            node for node in self.nodes.values()
            if node.branch_root == branch_root_id
        ]

    def get_depth_statistics(self) -> Dict[int, int]:
        """Retourne les statistiques de profondeur fractale.

        Returns:
            Dict mapping depth -> nombre de noeuds à cette profondeur
        """
        stats: Dict[int, int] = {}
        for node in self.nodes.values():
            stats[node.depth] = stats.get(node.depth, 0) + 1
        return stats

    def get_best_node(self) -> Optional[DAGNode]:
        """Retourne le noeud avec le meilleur score."""
        if self.best_node_id is None:
            return None
        return self.nodes[self.best_node_id]

    def get_node_states(self, node_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Récupère les états y et z d'un noeud pour le backtracking.

        Args:
            node_id: Identifiant du noeud

        Returns:
            Tuple (y_state, z_state) ou None si le noeud n'existe pas ou
            si les états n'ont pas été stockés
        """
        node = self.nodes.get(node_id)
        if node is None or node.y_state is None or node.z_state is None:
            return None
        return node.y_state, node.z_state

    def get_fractal_path(
        self,
        node_id: str,
        parent_index: int = 0
    ) -> List[DAGNode]:
        """Retourne le chemin fractal complet du noeud jusqu'à la racine.

        Traverse les branches et les parents pour reconstruire
        le chemin complet dans la structure fractale.

        Note: Dans un vrai DAG avec plusieurs parents, cette méthode suit
        un seul chemin (spécifié par parent_index). Pour une traversée
        complète de tous les chemins, utilisez une traversée BFS/DFS.

        Args:
            node_id: ID du noeud de départ
            parent_index: Index du parent à suivre quand plusieurs parents
                          existent (défaut: 0 = premier parent)

        Returns:
            Liste des noeuds du chemin (de la racine au noeud)
        """
        path: List[DAGNode] = []
        current_id: Optional[str] = node_id

        while current_id is not None:
            node = self.nodes.get(current_id)
            if node is None:
                break
            path.append(node)
            # Remonter vers le parent spécifié (ou None si pas de parents)
            if node.parents and len(node.parents) > parent_index:
                current_id = node.parents[parent_index]
            elif node.parents:
                current_id = node.parents[0]  # Fallback au premier parent
            else:
                current_id = None

        return list(reversed(path))


# ============================
#  Modèle complet T-RLINKOS TRM++ (NumPy)
# ============================

class TRLinkosTRM:
    """T-RLINKOS : Tiny Recursive Model ++ (NumPy version)

    Architecture récursive pour le raisonnement:
    - Coeur TRM (TRLinkosCore) avec experts dCaAP et routeur Torque
    - Merkle-DAG pour tracer et auditer le raisonnement
    - Backtracking optionnel vers les meilleurs états
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden_dim: int = 256,
        num_experts: int = 4,
    ) -> None:
        self.core = TRLinkosCore(x_dim, y_dim, z_dim, hidden_dim, num_experts)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        # encodeur simple pour x
        self.x_encoder = LinearNP(x_dim, x_dim)

        # y_init et z_init (paramètres globaux)
        self.y_init = np.zeros((1, y_dim), dtype=np.float64)
        self.z_init = np.zeros((1, z_dim), dtype=np.float64)

    def forward_recursive(
        self,
        x: np.ndarray,
        max_steps: int = 16,
        inner_recursions: int = 3,
        scorer: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        backtrack: bool = False,
        backtrack_threshold: float = 0.1,
    ) -> Tuple[np.ndarray, FractalMerkleDAG]:
        """Boucle de raisonnement récursif complète.

        Args:
            x: Entrée [B, dx]
            max_steps: Nombre maximal d'étapes de raisonnement
            inner_recursions: Nombre de récursions internes par étape
            scorer: Fonction optionnelle (x, y) -> score [B] (plus haut = mieux)
            backtrack: Si True, active le backtracking vers les meilleurs états
            backtrack_threshold: Seuil de dégradation du score pour déclencher
                                 le backtracking (fraction du meilleur score)

        Returns:
            y_final: Réponse finale [B, dy]
            dag: Merkle-DAG contenant l'historique du raisonnement
        """
        if x.ndim != 2 or x.shape[1] != self.x_dim:
            raise ValueError(f"x doit être de forme [B, {self.x_dim}], reçu {x.shape}")

        B = x.shape[0]
        x_enc = self.x_encoder(x)

        y = np.repeat(self.y_init, B, axis=0)
        z = np.repeat(self.z_init, B, axis=0)

        # Stocker les états si backtracking activé
        dag = FractalMerkleDAG(store_states=backtrack)
        current_node_ids: List[Optional[str]] = [None] * B

        # Meilleurs scores et états par échantillon (pour backtracking)
        best_scores_per_sample: List[float] = [float("-inf")] * B
        best_node_ids_per_sample: List[Optional[str]] = [None] * B

        for step in range(max_steps):
            y_next, z_next = self.core.step_reasoning(
                x_enc, y, z, inner_recursions=inner_recursions
            )

            # Score de la réponse si un scorer est fourni
            if scorer is not None:
                score_tensor = scorer(x, y_next)  # [B]
                scores = list(score_tensor)
            else:
                scores = [None] * B

            new_node_ids: List[str] = []
            for i in range(B):
                parents: List[str] = []
                if current_node_ids[i] is not None:
                    parents.append(current_node_ids[i])

                node_id = dag.add_step(
                    step=step,
                    y=y_next[i:i + 1],
                    z=z_next[i:i + 1],
                    parents=parents,
                    score=scores[i],
                )
                new_node_ids.append(node_id)

                # Suivre les meilleurs scores par échantillon
                if scores[i] is not None and scores[i] > best_scores_per_sample[i]:
                    best_scores_per_sample[i] = scores[i]
                    best_node_ids_per_sample[i] = node_id

            current_node_ids = new_node_ids
            y, z = y_next, z_next

            # Backtracking: si le score actuel est significativement pire
            # que le meilleur score vu, revenir au meilleur état
            if backtrack and scorer is not None and step < max_steps - 1:
                for i in range(B):
                    current_score = scores[i]
                    best_score = best_scores_per_sample[i]

                    # Vérifier si le score s'est dégradé significativement
                    should_backtrack = (
                        current_score is not None
                        and best_score > float("-inf")
                        and best_node_ids_per_sample[i] is not None
                    )

                    if should_backtrack:
                        # Calcul du seuil de dégradation
                        score_drop = best_score - current_score
                        threshold = abs(best_score) * backtrack_threshold

                        if score_drop > threshold:
                            # Restaurer l'état du meilleur noeud
                            states = dag.get_node_states(best_node_ids_per_sample[i])
                            if states is not None:
                                y_restored, z_restored = states
                                y[i:i + 1] = y_restored
                                z[i:i + 1] = z_restored
                                # Mettre à jour le parent pour le prochain step
                                current_node_ids[i] = best_node_ids_per_sample[i]

        # Retourner la meilleure prédiction si backtracking activé
        if backtrack and scorer is not None:
            for i in range(B):
                if best_node_ids_per_sample[i] is not None:
                    states = dag.get_node_states(best_node_ids_per_sample[i])
                    if states is not None:
                        y_best, _ = states
                        y[i:i + 1] = y_best

        return y, dag

    def forward_recursive_fractal(
        self,
        x: np.ndarray,
        max_steps: int = 16,
        inner_recursions: int = 3,
        scorer: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        backtrack: bool = False,
        backtrack_threshold: float = 0.1,
        fractal_branching: bool = True,
        branch_threshold: float = 0.05,
        max_branches_per_node: int = 2,
        perturbation_scale: float = 0.1,
    ) -> Tuple[np.ndarray, FractalMerkleDAG]:
        """Boucle de raisonnement récursif complète avec exploration fractale.

        Cette méthode étend forward_recursive en intégrant la création de
        branches fractales (sous-DAGs) pour explorer des alternatives
        pendant le raisonnement. Cela permet une exploration plus riche
        de l'espace des solutions.

        Args:
            x: Entrée [B, dx]
            max_steps: Nombre maximal d'étapes de raisonnement
            inner_recursions: Nombre de récursions internes par étape
            scorer: Fonction optionnelle (x, y) -> score [B] (plus haut = mieux)
            backtrack: Si True, active le backtracking vers les meilleurs états
            backtrack_threshold: Seuil de dégradation du score pour déclencher
                                 le backtracking (fraction du meilleur score)
            fractal_branching: Si True, crée des branches fractales pour explorer
                               des alternatives
            branch_threshold: Seuil de variabilité du score pour créer une branche
            max_branches_per_node: Nombre maximum de branches par noeud
            perturbation_scale: Échelle de perturbation pour les branches (adapté au
                                domaine du problème et à l'échelle du modèle)

        Returns:
            y_final: Réponse finale [B, dy]
            dag: Merkle-DAG fractal contenant l'historique du raisonnement
        """
        if x.ndim != 2 or x.shape[1] != self.x_dim:
            raise ValueError(f"x doit être de forme [B, {self.x_dim}], reçu {x.shape}")

        B = x.shape[0]
        x_enc = self.x_encoder(x)

        y = np.repeat(self.y_init, B, axis=0)
        z = np.repeat(self.z_init, B, axis=0)

        # Stocker les états pour backtracking et branches fractales
        dag = FractalMerkleDAG(
            store_states=backtrack or fractal_branching,
            max_depth=3 if fractal_branching else 1
        )
        current_node_ids: List[Optional[str]] = [None] * B

        # Meilleurs scores et états par échantillon (pour backtracking)
        best_scores_per_sample: List[float] = [float("-inf")] * B
        best_node_ids_per_sample: List[Optional[str]] = [None] * B

        # Historique des scores pour détecter la variabilité (branches fractales)
        score_history: List[List[float]] = [[] for _ in range(B)]

        for step in range(max_steps):
            y_next, z_next = self.core.step_reasoning(
                x_enc, y, z, inner_recursions=inner_recursions
            )

            # Score de la réponse si un scorer est fourni
            if scorer is not None:
                score_tensor = scorer(x, y_next)  # [B]
                scores = list(score_tensor)
            else:
                scores = [None] * B

            new_node_ids: List[str] = []
            for i in range(B):
                parents: List[str] = []
                if current_node_ids[i] is not None:
                    parents.append(current_node_ids[i])

                node_id = dag.add_step(
                    step=step,
                    y=y_next[i:i + 1],
                    z=z_next[i:i + 1],
                    parents=parents,
                    score=scores[i],
                )
                new_node_ids.append(node_id)

                # Suivre les meilleurs scores par échantillon
                if scores[i] is not None:
                    score_history[i].append(scores[i])
                    if scores[i] > best_scores_per_sample[i]:
                        best_scores_per_sample[i] = scores[i]
                        best_node_ids_per_sample[i] = node_id

                # Exploration fractale: créer des branches alternatives
                if (fractal_branching and scorer is not None
                        and len(score_history[i]) >= 2
                        and current_node_ids[i] is not None):

                    # Calculer la variabilité du score
                    recent_scores = score_history[i][-3:]
                    score_variance = np.var(recent_scores) if len(recent_scores) >= 2 else 0.0

                    # Créer une branche si la variabilité est élevée
                    # (exploration de l'espace des solutions)
                    if score_variance > branch_threshold:
                        # Compter les branches existantes du noeud parent
                        parent_node = dag.nodes.get(current_node_ids[i])
                        if parent_node is not None:
                            existing_branches = sum(
                                1 for child_id in parent_node.children
                                if dag.nodes.get(child_id) is not None
                                and dag.nodes[child_id].depth > parent_node.depth
                            )

                            if existing_branches < max_branches_per_node:
                                # Perturber légèrement y et z pour explorer une alternative
                                y_perturbed = y_next[i:i + 1] + np.random.randn(
                                    1, self.y_dim
                                ) * perturbation_scale
                                z_perturbed = z_next[i:i + 1] + np.random.randn(
                                    1, self.z_dim
                                ) * perturbation_scale

                                # Scorer la branche perturbée
                                branch_score = scorer(
                                    x[i:i + 1], y_perturbed
                                )[0]

                                # Créer la branche fractale
                                dag.create_branch(
                                    parent_node_id=current_node_ids[i],
                                    y=y_perturbed,
                                    z=z_perturbed,
                                    score=branch_score,
                                )

            current_node_ids = new_node_ids
            y, z = y_next, z_next

            # Backtracking: si le score actuel est significativement pire
            # que le meilleur score vu, revenir au meilleur état
            if backtrack and scorer is not None and step < max_steps - 1:
                for i in range(B):
                    current_score = scores[i]
                    best_score = best_scores_per_sample[i]

                    # Vérifier si le score s'est dégradé significativement
                    should_backtrack = (
                        current_score is not None
                        and best_score > float("-inf")
                        and best_node_ids_per_sample[i] is not None
                    )

                    if should_backtrack:
                        # Calcul du seuil de dégradation
                        score_drop = best_score - current_score
                        threshold = abs(best_score) * backtrack_threshold

                        if score_drop > threshold:
                            # Restaurer l'état du meilleur noeud
                            states = dag.get_node_states(best_node_ids_per_sample[i])
                            if states is not None:
                                y_restored, z_restored = states
                                y[i:i + 1] = y_restored
                                z[i:i + 1] = z_restored
                                # Mettre à jour le parent pour le prochain step
                                current_node_ids[i] = best_node_ids_per_sample[i]

        # Retourner la meilleure prédiction si backtracking activé
        if backtrack and scorer is not None:
            for i in range(B):
                if best_node_ids_per_sample[i] is not None:
                    states = dag.get_node_states(best_node_ids_per_sample[i])
                    if states is not None:
                        y_best, _ = states
                        y[i:i + 1] = y_best

        return y, dag


# ============================
#  Data Encoders for Text and Images
# ============================


class TextEncoder:
    """Encodeur simple pour les données textuelles.

    Convertit le texte en représentations vectorielles via:
    - Tokenisation caractère ou mot
    - Embedding appris
    - Agrégation (moyenne, max, ou attention simplifiée)

    Note: Cette implémentation est un encodeur basique en NumPy.
    Pour la production, utiliser des modèles pré-entraînés (BERT, etc.)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        output_dim: int = 64,
        mode: str = "char"
    ):
        """Initialise l'encodeur de texte.

        Args:
            vocab_size: Taille du vocabulaire (256 pour caractères ASCII)
            embed_dim: Dimension des embeddings
            output_dim: Dimension de sortie
            mode: "char" pour caractères, "word" pour mots
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.mode = mode

        # Table d'embedding (initialisée aléatoirement)
        limit = np.sqrt(2.0 / embed_dim)
        self.embedding_table = np.random.uniform(
            -limit, limit, (vocab_size, embed_dim)
        )

        # Projection vers output_dim
        self.output_projection = LinearNP(embed_dim, output_dim)

        # Dictionnaire mot -> index (pour mode word)
        self.word_to_idx: Dict[str, int] = {}
        self.next_idx = 0

    def _tokenize_char(self, text: str) -> List[int]:
        """Tokenisation par caractère."""
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def _tokenize_word(self, text: str) -> List[int]:
        """Tokenisation par mot."""
        words = text.lower().split()
        tokens = []
        for word in words:
            if word not in self.word_to_idx:
                if self.next_idx < self.vocab_size:
                    self.word_to_idx[word] = self.next_idx
                    self.next_idx += 1
                else:
                    # Deterministic hash using hashlib for vocabulary overflow
                    word_hash = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
                    self.word_to_idx[word] = word_hash % self.vocab_size
            tokens.append(self.word_to_idx[word])
        return tokens

    def encode(self, texts: List[str], max_length: int = 128) -> np.ndarray:
        """Encode une liste de textes en vecteurs.

        Args:
            texts: Liste de textes à encoder
            max_length: Longueur maximale de la séquence

        Returns:
            Représentations vectorielles [B, output_dim]
        """
        batch_embeddings = []

        for text in texts:
            # Tokenisation
            if self.mode == "char":
                tokens = self._tokenize_char(text)
            else:
                tokens = self._tokenize_word(text)

            # Tronquer ou padder
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))

            # Lookup embeddings
            embeddings = self.embedding_table[tokens]  # [seq_len, embed_dim]

            # Agrégation (moyenne)
            # Masquer les tokens de padding
            mask = np.array(tokens) != 0
            if mask.sum() > 0:
                aggregated = np.sum(
                    embeddings * mask[:, None], axis=0
                ) / max(mask.sum(), 1)
            else:
                aggregated = np.zeros(self.embed_dim)

            batch_embeddings.append(aggregated)

        # Stack et projection
        batch_tensor = np.stack(batch_embeddings, axis=0)  # [B, embed_dim]
        output = self.output_projection(batch_tensor)  # [B, output_dim]
        return output


class ImageEncoder:
    """Encodeur simple pour les données d'images.

    Convertit les images en représentations vectorielles via:
    - Convolution simplifiée (patches)
    - Pooling et projection

    Note: Cette implémentation est un encodeur basique en NumPy.
    Pour la production, utiliser des modèles pré-entraînés (ResNet, ViT, etc.)
    """

    def __init__(
        self,
        input_channels: int = 3,
        patch_size: int = 8,
        embed_dim: int = 64,
        output_dim: int = 64
    ):
        """Initialise l'encodeur d'image.

        Args:
            input_channels: Nombre de canaux d'entrée (3 pour RGB, 1 pour grayscale)
            patch_size: Taille des patches pour la "convolution"
            embed_dim: Dimension des embeddings de patch
            output_dim: Dimension de sortie
        """
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Projection linéaire des patches vers embed_dim
        patch_dim = patch_size * patch_size * input_channels
        self.patch_projection = LinearNP(patch_dim, embed_dim)

        # Projection vers output_dim
        self.output_projection = LinearNP(embed_dim, output_dim)

    def _extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Extrait les patches d'une image.

        Args:
            image: Image [H, W, C] ou [H, W] pour grayscale

        Returns:
            Patches aplatis [num_patches, patch_dim]
        """
        # Normaliser à [H, W, C]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        H, W, C = image.shape
        ps = self.patch_size

        # Calculer le nombre de patches
        num_patches_h = H // ps
        num_patches_w = W // ps

        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = image[i*ps:(i+1)*ps, j*ps:(j+1)*ps, :]
                patches.append(patch.flatten())

        if len(patches) == 0:
            # Image trop petite: pad to minimum patch size and use
            patch_dim = self.patch_size * self.patch_size * self.input_channels
            # Flatten and pad the small image
            flat_image = image.flatten()
            if len(flat_image) < patch_dim:
                padded = np.zeros(patch_dim)
                padded[:len(flat_image)] = flat_image
                return padded.reshape(1, -1)
            else:
                return flat_image[:patch_dim].reshape(1, -1)

        return np.stack(patches, axis=0)

    def encode(self, images: List[np.ndarray]) -> np.ndarray:
        """Encode une liste d'images en vecteurs.

        Args:
            images: Liste d'images (numpy arrays)

        Returns:
            Représentations vectorielles [B, output_dim]
        """
        batch_embeddings = []

        for image in images:
            # Normaliser l'image entre 0 et 1
            if image.max() > 1:
                image = image.astype(np.float64) / 255.0

            # Extraire les patches
            patches = self._extract_patches(image)  # [num_patches, patch_dim]

            # Projeter les patches
            patch_embeddings = self.patch_projection(patches)  # [num_patches, embed_dim]

            # Agrégation (moyenne des patches)
            aggregated = np.mean(patch_embeddings, axis=0)  # [embed_dim]

            batch_embeddings.append(aggregated)

        # Stack et projection
        batch_tensor = np.stack(batch_embeddings, axis=0)  # [B, embed_dim]
        output = self.output_projection(batch_tensor)  # [B, output_dim]
        return output


# ============================
#  Dataset and DataLoader utilities
# ============================


@dataclass
class DataSample:
    """Échantillon de données pour l'entraînement.

    Attributes:
        x: Données d'entrée (déjà encodées) [x_dim]
        y_target: Cible [y_dim]
        raw_data: Données brutes optionnelles (texte, image, etc.)
        metadata: Métadonnées optionnelles
    """
    x: np.ndarray
    y_target: np.ndarray
    raw_data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class Dataset:
    """Dataset simple pour T-RLINKOS.

    Gère les données d'entraînement avec support pour:
    - Données vectorielles brutes
    - Texte (via TextEncoder)
    - Images (via ImageEncoder)
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        encoder_type: str = "vector",
        text_encoder: Optional[TextEncoder] = None,
        image_encoder: Optional[ImageEncoder] = None
    ):
        """Initialise le dataset.

        Args:
            x_dim: Dimension d'entrée attendue
            y_dim: Dimension de sortie attendue
            encoder_type: "vector", "text", ou "image"
            text_encoder: Encodeur de texte (si encoder_type="text")
            image_encoder: Encodeur d'image (si encoder_type="image")
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.encoder_type = encoder_type
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.samples: List[DataSample] = []

    def add_sample(
        self,
        x: Union[np.ndarray, str, Any],
        y_target: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Ajoute un échantillon au dataset.

        Args:
            x: Entrée (vecteur, texte, ou image selon encoder_type)
            y_target: Cible [y_dim]
            metadata: Métadonnées optionnelles
        """
        raw_data = x

        # Encoder selon le type
        if self.encoder_type == "text" and isinstance(x, str):
            if self.text_encoder is None:
                raise ValueError("TextEncoder requis pour encoder_type='text'")
            x_encoded = self.text_encoder.encode([x])[0]
        elif self.encoder_type == "image" and isinstance(x, np.ndarray) and x.ndim >= 2:
            if self.image_encoder is None:
                raise ValueError("ImageEncoder requis pour encoder_type='image'")
            x_encoded = self.image_encoder.encode([x])[0]
        else:
            x_encoded = np.asarray(x).flatten()
            if len(x_encoded) != self.x_dim:
                # Padder ou tronquer si nécessaire
                if len(x_encoded) < self.x_dim:
                    x_encoded = np.pad(x_encoded, (0, self.x_dim - len(x_encoded)))
                else:
                    x_encoded = x_encoded[:self.x_dim]

        y_target = np.asarray(y_target).flatten()
        if len(y_target) != self.y_dim:
            if len(y_target) < self.y_dim:
                y_target = np.pad(y_target, (0, self.y_dim - len(y_target)))
            else:
                y_target = y_target[:self.y_dim]

        sample = DataSample(
            x=x_encoded,
            y_target=y_target,
            raw_data=raw_data,
            metadata=metadata
        )
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DataSample:
        return self.samples[idx]


class DataLoader:
    """DataLoader simple pour l'entraînement par batches.

    Fournit des itérateurs sur les batches de données avec:
    - Shuffle optionnel
    - Batching configurable
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """Initialise le DataLoader.

        Args:
            dataset: Dataset à charger
            batch_size: Taille des batches
            shuffle: Si True, mélange les données à chaque époque
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Itère sur les batches."""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            x_batch = np.stack([self.dataset[i].x for i in batch_indices], axis=0)
            y_batch = np.stack([self.dataset[i].y_target for i in batch_indices], axis=0)
            yield x_batch, y_batch

    def __len__(self) -> int:
        """Retourne le nombre de batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ============================
#  Loss Functions
# ============================


def mse_loss(y_pred: np.ndarray, y_target: np.ndarray) -> float:
    """Mean Squared Error loss.

    Args:
        y_pred: Prédictions [B, y_dim]
        y_target: Cibles [B, y_dim]

    Returns:
        Loss scalaire
    """
    return float(np.mean((y_pred - y_target) ** 2))


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """Cross-entropy loss pour classification.

    Args:
        logits: Logits [B, num_classes]
        targets: Indices de classes [B] ou one-hot [B, num_classes]
        epsilon: Petit nombre pour stabilité numérique

    Returns:
        Loss scalaire
    """
    # Convertir en probabilités
    probs = softmax(logits, axis=-1)

    # Si targets sont des indices, convertir en one-hot
    if targets.ndim == 1:
        num_classes = logits.shape[-1]
        targets_onehot = np.zeros_like(logits)
        targets_onehot[np.arange(len(targets)), targets.astype(int)] = 1
        targets = targets_onehot

    # Cross-entropy
    return float(-np.mean(np.sum(targets * np.log(probs + epsilon), axis=-1)))


def cosine_similarity_loss(y_pred: np.ndarray, y_target: np.ndarray) -> float:
    """Cosine similarity loss (1 - cosine_similarity).

    Args:
        y_pred: Prédictions [B, y_dim]
        y_target: Cibles [B, y_dim]

    Returns:
        Loss scalaire
    """
    # Normaliser
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-10)
    y_target_norm = y_target / (np.linalg.norm(y_target, axis=-1, keepdims=True) + 1e-10)

    # Similarité cosinus
    cos_sim = np.sum(y_pred_norm * y_target_norm, axis=-1)

    return float(1 - np.mean(cos_sim))


# ============================
#  Training Pipeline
# ============================


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement.

    Attributes:
        learning_rate: Taux d'apprentissage
        num_epochs: Nombre d'époques
        batch_size: Taille des batches
        max_steps: Nombre maximal d'étapes de raisonnement
        inner_recursions: Nombre de récursions internes
        use_fractal_branching: Active l'exploration fractale
        loss_fn: Fonction de loss ("mse", "cross_entropy", "cosine")
        log_interval: Intervalle d'affichage des logs
        gradient_clip: Valeur maximale du gradient (si > 0)
    """
    learning_rate: float = 0.01
    num_epochs: int = 100
    batch_size: int = 32
    max_steps: int = 8
    inner_recursions: int = 3
    use_fractal_branching: bool = False
    loss_fn: str = "mse"
    log_interval: int = 10
    gradient_clip: float = 1.0


class Trainer:
    """Pipeline d'entraînement pour T-RLINKOS.

    Implémente:
    - Boucle d'entraînement avec données réelles
    - Calcul de gradients numériques (finite differences)
    - Mise à jour des paramètres
    - Logging et métriques
    """

    def __init__(
        self,
        model: TRLinkosTRM,
        config: TrainingConfig
    ):
        """Initialise le trainer.

        Args:
            model: Modèle T-RLINKOS à entraîner
            config: Configuration d'entraînement
        """
        self.model = model
        self.config = config

        # Sélectionner la fonction de loss
        self.loss_functions = {
            "mse": mse_loss,
            "cross_entropy": cross_entropy_loss,
            "cosine": cosine_similarity_loss,
        }
        self.loss_fn = self.loss_functions.get(config.loss_fn, mse_loss)

        # Historique d'entraînement
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "epoch": [],
        }

    def _collect_parameters(self) -> List[Tuple[np.ndarray, str]]:
        """Collecte tous les paramètres du modèle.

        Returns:
            Liste de tuples (paramètre, chemin)
        """
        params = []

        # Encodeur x
        params.append((self.model.x_encoder.W, "x_encoder.W"))
        params.append((self.model.x_encoder.b, "x_encoder.b"))

        # y_init et z_init
        params.append((self.model.y_init, "y_init"))
        params.append((self.model.z_init, "z_init"))

        # Core: experts
        for e_idx, expert in enumerate(self.model.core.experts):
            for b_idx, branch_w in enumerate(expert.branch_weights):
                params.append((branch_w.W, f"expert_{e_idx}.branch_{b_idx}.W"))
                params.append((branch_w.b, f"expert_{e_idx}.branch_{b_idx}.b"))
            params.append((expert.soma_integration.W, f"expert_{e_idx}.soma.W"))
            params.append((expert.soma_integration.b, f"expert_{e_idx}.soma.b"))
            params.append((expert.calcium_gate.W, f"expert_{e_idx}.ca_gate.W"))
            params.append((expert.calcium_gate.b, f"expert_{e_idx}.ca_gate.b"))
            params.append((expert.output_projection.W, f"expert_{e_idx}.output.W"))
            params.append((expert.output_projection.b, f"expert_{e_idx}.output.b"))

        # Core: router
        params.append((self.model.core.router.projection.W, "router.proj.W"))
        params.append((self.model.core.router.projection.b, "router.proj.b"))
        params.append((self.model.core.router.expert_centroids, "router.centroids"))
        params.append((self.model.core.router.mass_projection.W, "router.mass.W"))
        params.append((self.model.core.router.mass_projection.b, "router.mass.b"))

        # Core: answer dense
        params.append((self.model.core.answer_dense1.W, "answer1.W"))
        params.append((self.model.core.answer_dense1.b, "answer1.b"))
        params.append((self.model.core.answer_dense2.W, "answer2.W"))
        params.append((self.model.core.answer_dense2.b, "answer2.b"))

        return params

    def _compute_loss(
        self,
        x: np.ndarray,
        y_target: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Calcule la loss et la prédiction.

        Args:
            x: Entrées [B, x_dim]
            y_target: Cibles [B, y_dim]

        Returns:
            (loss, y_pred)
        """
        if self.config.use_fractal_branching:
            y_pred, _ = self.model.forward_recursive_fractal(
                x,
                max_steps=self.config.max_steps,
                inner_recursions=self.config.inner_recursions,
            )
        else:
            y_pred, _ = self.model.forward_recursive(
                x,
                max_steps=self.config.max_steps,
                inner_recursions=self.config.inner_recursions,
            )

        loss = self.loss_fn(y_pred, y_target)
        return loss, y_pred

    def _compute_gradient_numeric(
        self,
        param: np.ndarray,
        x: np.ndarray,
        y_target: np.ndarray,
        epsilon: float = 1e-5,
        sample_ratio: float = 0.1,
        min_samples: int = 5,
        max_samples: int = 50
    ) -> np.ndarray:
        """Calcule le gradient numérique par différences finies.

        Args:
            param: Paramètre pour lequel calculer le gradient
            x: Entrées [B, x_dim]
            y_target: Cibles [B, y_dim]
            epsilon: Pas pour les différences finies
            sample_ratio: Ratio de paramètres à échantillonner (0.0 à 1.0)
            min_samples: Nombre minimum de paramètres à échantillonner
            max_samples: Nombre maximum de paramètres à échantillonner

        Returns:
            Gradient de même forme que param
        """
        grad = np.zeros_like(param)
        flat_param = param.ravel()
        flat_grad = grad.ravel()

        # Échantillonner un sous-ensemble d'indices pour efficacité
        # Adapter le nombre d'échantillons à la taille du paramètre
        num_from_ratio = int(len(flat_param) * sample_ratio)
        num_samples = max(min_samples, min(num_from_ratio, max_samples))
        num_samples = min(num_samples, len(flat_param))

        sampled_indices = np.random.choice(
            len(flat_param), num_samples, replace=False
        )

        for idx in sampled_indices:
            original_value = flat_param[idx]

            # f(x + epsilon)
            flat_param[idx] = original_value + epsilon
            loss_plus, _ = self._compute_loss(x, y_target)

            # f(x - epsilon)
            flat_param[idx] = original_value - epsilon
            loss_minus, _ = self._compute_loss(x, y_target)

            # Restaurer
            flat_param[idx] = original_value

            # Gradient par différences centrées
            flat_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        return grad

    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Clip le gradient si nécessaire.

        Args:
            grad: Gradient à clipper

        Returns:
            Gradient clippé
        """
        if self.config.gradient_clip > 0:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.gradient_clip:
                grad = grad * self.config.gradient_clip / grad_norm
        return grad

    def train_epoch(
        self,
        dataloader: DataLoader
    ) -> float:
        """Effectue une époque d'entraînement.

        Args:
            dataloader: DataLoader des données d'entraînement

        Returns:
            Loss moyenne de l'époque
        """
        epoch_losses = []
        params = self._collect_parameters()

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            # Calculer la loss
            loss, _ = self._compute_loss(x_batch, y_batch)
            epoch_losses.append(loss)

            # Calculer et appliquer les gradients
            for param, name in params:
                grad = self._compute_gradient_numeric(param, x_batch, y_batch)
                grad = self._clip_gradient(grad)

                # Mise à jour SGD
                param -= self.config.learning_rate * grad

        return float(np.mean(epoch_losses))

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> Dict[str, List[float]]:
        """Boucle d'entraînement complète.

        Args:
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation (optionnel)

        Returns:
            Historique d'entraînement
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            self.history["val_loss"] = []

        print(f"Démarrage de l'entraînement: {self.config.num_epochs} époques")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Max steps: {self.config.max_steps}")
        print(f"  - Fractal branching: {self.config.use_fractal_branching}")
        print("-" * 50)

        for epoch in range(self.config.num_epochs):
            # Entraînement
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["epoch"].append(epoch)

            # Validation
            if val_loader is not None:
                val_losses = []
                for x_batch, y_batch in val_loader:
                    loss, _ = self._compute_loss(x_batch, y_batch)
                    val_losses.append(loss)
                val_loss = float(np.mean(val_losses))
                self.history["val_loss"].append(val_loss)

            # Logging
            if epoch % self.config.log_interval == 0 or epoch == self.config.num_epochs - 1:
                log_msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f}"
                if val_loader is not None:
                    log_msg += f" | Val Loss: {val_loss:.6f}"
                print(log_msg)

        print("-" * 50)
        print("Entraînement terminé!")

        return self.history

    def evaluate(
        self,
        dataset: Dataset
    ) -> Tuple[float, np.ndarray]:
        """Évalue le modèle sur un dataset.

        Args:
            dataset: Dataset à évaluer

        Returns:
            (loss, predictions)
        """
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        losses = []
        predictions = []

        for x_batch, y_batch in loader:
            loss, y_pred = self._compute_loss(x_batch, y_batch)
            losses.append(loss)
            predictions.append(y_pred)

        total_loss = float(np.mean(losses))
        all_predictions = np.concatenate(predictions, axis=0)

        return total_loss, all_predictions


# ============================
#  Model Serialization (Phase 1)
# ============================


def save_model(model: TRLinkosTRM, filepath: str) -> None:
    """Save a T-RLINKOS model to disk.

    Serializes all model parameters using NumPy's native format.
    The saved file can be loaded with `load_model()`.

    Args:
        model: The TRLinkosTRM model to save
        filepath: Path to save the model (recommended extension: .npz)

    Example:
        >>> model = TRLinkosTRM(64, 32, 64)
        >>> save_model(model, "my_model.npz")
    """
    params = _collect_model_params(model)
    config = {
        "x_dim": model.x_dim,
        "y_dim": model.y_dim,
        "z_dim": model.z_dim,
        "hidden_dim": model.core.answer_dense1.W.shape[0],
        "num_experts": model.core.num_experts,
    }

    # Save parameters and config together
    save_dict = {f"param_{k}": v for k, v in params.items()}
    save_dict["config_x_dim"] = np.array([config["x_dim"]])
    save_dict["config_y_dim"] = np.array([config["y_dim"]])
    save_dict["config_z_dim"] = np.array([config["z_dim"]])
    save_dict["config_hidden_dim"] = np.array([config["hidden_dim"]])
    save_dict["config_num_experts"] = np.array([config["num_experts"]])

    np.savez_compressed(filepath, **save_dict)


def load_model(filepath: str) -> TRLinkosTRM:
    """Load a T-RLINKOS model from disk.

    Loads a model saved with `save_model()`.

    Args:
        filepath: Path to the saved model file

    Returns:
        Loaded TRLinkosTRM model with restored parameters

    Example:
        >>> model = load_model("my_model.npz")
        >>> y_pred, dag = model.forward_recursive(x_batch)
    """
    data = np.load(filepath, allow_pickle=False)

    # Extract config
    x_dim = int(data["config_x_dim"][0])
    y_dim = int(data["config_y_dim"][0])
    z_dim = int(data["config_z_dim"][0])
    hidden_dim = int(data["config_hidden_dim"][0])
    num_experts = int(data["config_num_experts"][0])

    # Create model with same architecture
    model = TRLinkosTRM(x_dim, y_dim, z_dim, hidden_dim, num_experts)

    # Restore parameters
    params = {k.replace("param_", ""): data[k] for k in data.files if k.startswith("param_")}
    _restore_model_params(model, params)

    return model


def _collect_model_params(model: TRLinkosTRM) -> Dict[str, np.ndarray]:
    """Collect all model parameters into a dictionary.

    Args:
        model: TRLinkosTRM model

    Returns:
        Dictionary mapping parameter names to numpy arrays
    """
    params = {}

    # x_encoder
    params["x_encoder_W"] = model.x_encoder.W
    params["x_encoder_b"] = model.x_encoder.b

    # y_init and z_init
    params["y_init"] = model.y_init
    params["z_init"] = model.z_init

    # Core: experts
    for e_idx, expert in enumerate(model.core.experts):
        for b_idx, branch_w in enumerate(expert.branch_weights):
            params[f"expert_{e_idx}_branch_{b_idx}_W"] = branch_w.W
            params[f"expert_{e_idx}_branch_{b_idx}_b"] = branch_w.b
        for b_idx, threshold in enumerate(expert.branch_thresholds):
            params[f"expert_{e_idx}_threshold_{b_idx}"] = threshold
        params[f"expert_{e_idx}_soma_W"] = expert.soma_integration.W
        params[f"expert_{e_idx}_soma_b"] = expert.soma_integration.b
        params[f"expert_{e_idx}_ca_gate_W"] = expert.calcium_gate.W
        params[f"expert_{e_idx}_ca_gate_b"] = expert.calcium_gate.b
        params[f"expert_{e_idx}_output_W"] = expert.output_projection.W
        params[f"expert_{e_idx}_output_b"] = expert.output_projection.b

    # Core: router
    params["router_proj_W"] = model.core.router.projection.W
    params["router_proj_b"] = model.core.router.projection.b
    params["router_centroids"] = model.core.router.expert_centroids
    params["router_mass_W"] = model.core.router.mass_projection.W
    params["router_mass_b"] = model.core.router.mass_projection.b

    # Core: answer dense
    params["answer1_W"] = model.core.answer_dense1.W
    params["answer1_b"] = model.core.answer_dense1.b
    params["answer2_W"] = model.core.answer_dense2.W
    params["answer2_b"] = model.core.answer_dense2.b

    return params


def _restore_model_params(model: TRLinkosTRM, params: Dict[str, np.ndarray]) -> None:
    """Restore model parameters from a dictionary.

    Args:
        model: TRLinkosTRM model to restore parameters into
        params: Dictionary mapping parameter names to numpy arrays
    """
    # x_encoder
    model.x_encoder.W[:] = params["x_encoder_W"]
    model.x_encoder.b[:] = params["x_encoder_b"]

    # y_init and z_init
    model.y_init[:] = params["y_init"]
    model.z_init[:] = params["z_init"]

    # Core: experts
    for e_idx, expert in enumerate(model.core.experts):
        for b_idx, branch_w in enumerate(expert.branch_weights):
            branch_w.W[:] = params[f"expert_{e_idx}_branch_{b_idx}_W"]
            branch_w.b[:] = params[f"expert_{e_idx}_branch_{b_idx}_b"]
        for b_idx in range(len(expert.branch_thresholds)):
            expert.branch_thresholds[b_idx][:] = params[f"expert_{e_idx}_threshold_{b_idx}"]
        expert.soma_integration.W[:] = params[f"expert_{e_idx}_soma_W"]
        expert.soma_integration.b[:] = params[f"expert_{e_idx}_soma_b"]
        expert.calcium_gate.W[:] = params[f"expert_{e_idx}_ca_gate_W"]
        expert.calcium_gate.b[:] = params[f"expert_{e_idx}_ca_gate_b"]
        expert.output_projection.W[:] = params[f"expert_{e_idx}_output_W"]
        expert.output_projection.b[:] = params[f"expert_{e_idx}_output_b"]

    # Core: router
    model.core.router.projection.W[:] = params["router_proj_W"]
    model.core.router.projection.b[:] = params["router_proj_b"]
    model.core.router.expert_centroids[:] = params["router_centroids"]
    model.core.router.mass_projection.W[:] = params["router_mass_W"]
    model.core.router.mass_projection.b[:] = params["router_mass_b"]

    # Core: answer dense
    model.core.answer_dense1.W[:] = params["answer1_W"]
    model.core.answer_dense1.b[:] = params["answer1_b"]
    model.core.answer_dense2.W[:] = params["answer2_W"]
    model.core.answer_dense2.b[:] = params["answer2_b"]


# ============================
#  Formal Benchmarks (Phase 1)
# ============================


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    Attributes:
        name: Name of the benchmark
        config: Configuration used (dimensions, etc.)
        total_time: Total execution time in seconds
        time_per_step: Average time per reasoning step
        time_per_sample: Average time per sample
        throughput: Samples per second
        memory_estimate_mb: Estimated memory usage in MB
        num_steps: Number of reasoning steps
        batch_size: Batch size used
    """
    name: str
    config: Dict[str, Any]
    total_time: float
    time_per_step: float
    time_per_sample: float
    throughput: float
    memory_estimate_mb: float
    num_steps: int
    batch_size: int


def benchmark_forward_recursive(
    model: TRLinkosTRM,
    batch_size: int = 32,
    max_steps: int = 16,
    inner_recursions: int = 3,
    num_runs: int = 5,
    warmup_runs: int = 2
) -> BenchmarkResult:
    """Benchmark the forward_recursive method.

    Measures execution time and estimates memory usage for the main
    inference loop.

    Args:
        model: TRLinkosTRM model to benchmark
        batch_size: Number of samples per batch
        max_steps: Maximum reasoning steps
        inner_recursions: Inner recursions per step
        num_runs: Number of timing runs (after warmup)
        warmup_runs: Number of warmup runs (not timed)

    Returns:
        BenchmarkResult with timing and throughput metrics
    """
    import time

    x_batch = np.random.randn(batch_size, model.x_dim)

    # Warmup runs
    for _ in range(warmup_runs):
        _ = model.forward_recursive(
            x_batch, max_steps=max_steps, inner_recursions=inner_recursions
        )

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.forward_recursive(
            x_batch, max_steps=max_steps, inner_recursions=inner_recursions
        )
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    time_per_step = avg_time / max_steps
    time_per_sample = avg_time / batch_size
    throughput = batch_size / avg_time

    # Memory estimate (rough calculation based on array sizes)
    # Count parameters in model
    params = _collect_model_params(model)
    param_bytes = sum(p.nbytes for p in params.values())

    # Add intermediate activations estimate
    # (batch_size * hidden_dim * max_steps * inner_recursions * num_experts)
    hidden_dim = model.core.answer_dense1.W.shape[0]
    num_experts = model.core.num_experts
    activation_estimate = (
        batch_size * hidden_dim * max_steps * inner_recursions * num_experts * 8  # float64
    )

    memory_mb = (param_bytes + activation_estimate) / (1024 * 1024)

    config = {
        "x_dim": model.x_dim,
        "y_dim": model.y_dim,
        "z_dim": model.z_dim,
        "hidden_dim": hidden_dim,
        "num_experts": num_experts,
    }

    return BenchmarkResult(
        name="forward_recursive",
        config=config,
        total_time=avg_time,
        time_per_step=time_per_step,
        time_per_sample=time_per_sample,
        throughput=throughput,
        memory_estimate_mb=memory_mb,
        num_steps=max_steps,
        batch_size=batch_size,
    )


def benchmark_forward_recursive_fractal(
    model: TRLinkosTRM,
    batch_size: int = 32,
    max_steps: int = 16,
    inner_recursions: int = 3,
    num_runs: int = 5,
    warmup_runs: int = 2
) -> BenchmarkResult:
    """Benchmark the forward_recursive_fractal method.

    Measures execution time and estimates memory usage for the fractal
    inference loop with branching.

    Args:
        model: TRLinkosTRM model to benchmark
        batch_size: Number of samples per batch
        max_steps: Maximum reasoning steps
        inner_recursions: Inner recursions per step
        num_runs: Number of timing runs (after warmup)
        warmup_runs: Number of warmup runs (not timed)

    Returns:
        BenchmarkResult with timing and throughput metrics
    """
    import time

    x_batch = np.random.randn(batch_size, model.x_dim)
    target = np.random.randn(batch_size, model.y_dim)

    def scorer(x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.mean((y_pred - target) ** 2, axis=-1)

    # Warmup runs
    for _ in range(warmup_runs):
        _ = model.forward_recursive_fractal(
            x_batch,
            max_steps=max_steps,
            inner_recursions=inner_recursions,
            scorer=scorer,
            fractal_branching=True,
        )

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.forward_recursive_fractal(
            x_batch,
            max_steps=max_steps,
            inner_recursions=inner_recursions,
            scorer=scorer,
            fractal_branching=True,
        )
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    time_per_step = avg_time / max_steps
    time_per_sample = avg_time / batch_size
    throughput = batch_size / avg_time

    # Memory estimate
    params = _collect_model_params(model)
    param_bytes = sum(p.nbytes for p in params.values())
    hidden_dim = model.core.answer_dense1.W.shape[0]
    num_experts = model.core.num_experts

    # Fractal branching uses more memory for DAG storage
    activation_estimate = (
        batch_size * hidden_dim * max_steps * inner_recursions * num_experts * 8 * 2
    )

    memory_mb = (param_bytes + activation_estimate) / (1024 * 1024)

    config = {
        "x_dim": model.x_dim,
        "y_dim": model.y_dim,
        "z_dim": model.z_dim,
        "hidden_dim": hidden_dim,
        "num_experts": num_experts,
    }

    return BenchmarkResult(
        name="forward_recursive_fractal",
        config=config,
        total_time=avg_time,
        time_per_step=time_per_step,
        time_per_sample=time_per_sample,
        throughput=throughput,
        memory_estimate_mb=memory_mb,
        num_steps=max_steps,
        batch_size=batch_size,
    )


def run_benchmark_suite(
    configs: Optional[List[Dict[str, int]]] = None,
    batch_sizes: Optional[List[int]] = None,
    max_steps: int = 8,
    num_runs: int = 3
) -> List[BenchmarkResult]:
    """Run a suite of benchmarks with different configurations.

    Args:
        configs: List of model configs (x_dim, y_dim, z_dim, hidden_dim, num_experts)
                 Default: small, medium, large configurations
        batch_sizes: List of batch sizes to test. Default: [1, 8, 32]
        max_steps: Maximum reasoning steps
        num_runs: Number of timing runs per benchmark

    Returns:
        List of BenchmarkResult objects

    Example:
        >>> results = run_benchmark_suite()
        >>> for r in results:
        ...     print(f"{r.name}: {r.throughput:.1f} samples/sec")
    """
    if configs is None:
        configs = [
            {"x_dim": 16, "y_dim": 8, "z_dim": 16, "hidden_dim": 64, "num_experts": 2},
            {"x_dim": 64, "y_dim": 32, "z_dim": 64, "hidden_dim": 256, "num_experts": 4},
            {"x_dim": 128, "y_dim": 64, "z_dim": 128, "hidden_dim": 512, "num_experts": 8},
        ]

    if batch_sizes is None:
        batch_sizes = [1, 8, 32]

    results = []

    for config in configs:
        model = TRLinkosTRM(**config)

        for batch_size in batch_sizes:
            # Benchmark forward_recursive
            result = benchmark_forward_recursive(
                model,
                batch_size=batch_size,
                max_steps=max_steps,
                num_runs=num_runs,
                warmup_runs=1,
            )
            results.append(result)

            # Benchmark forward_recursive_fractal
            result_fractal = benchmark_forward_recursive_fractal(
                model,
                batch_size=batch_size,
                max_steps=max_steps,
                num_runs=num_runs,
                warmup_runs=1,
            )
            results.append(result_fractal)

    return results


def print_benchmark_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table.

    Args:
        results: List of BenchmarkResult objects to display
    """
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)

    # Group by config
    current_config = None

    for r in results:
        config_str = f"x={r.config['x_dim']}, y={r.config['y_dim']}, z={r.config['z_dim']}, h={r.config['hidden_dim']}, E={r.config['num_experts']}"

        if config_str != current_config:
            if current_config is not None:
                print("-" * 90)
            print(f"\nConfiguration: {config_str}")
            print(f"{'Method':<30} {'Batch':>6} {'Time (ms)':>12} {'Throughput':>15} {'Memory (MB)':>12}")
            print("-" * 90)
            current_config = config_str

        print(
            f"{r.name:<30} {r.batch_size:>6} {r.total_time * 1000:>12.2f} "
            f"{r.throughput:>12.1f} s/s {r.memory_estimate_mb:>12.2f}"
        )

    print("=" * 90)


# ============================
#  Tests / Exemple d'utilisation minimal
# ============================

if __name__ == "__main__":
    np.random.seed(42)

    # --- Test 1 : dimensions d'origine ---
    x_dim, y_dim, z_dim = 64, 32, 64
    model = TRLinkosTRM(x_dim, y_dim, z_dim)

    # Batch de 8 entrées aléatoires
    x_batch = np.random.randn(8, x_dim)

    # Cible fictive pour illustrer un scorer
    target = np.random.randn(8, y_dim)

    def scorer(x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Score = -MSE (plus c'est proche de la cible, plus c'est élevé)
        return -np.mean((y_pred - target) ** 2, axis=-1)

    y_pred, dag = model.forward_recursive(
        x_batch,
        max_steps=10,
        inner_recursions=3,
        scorer=scorer,
        backtrack=False,
    )

    print("[Test 1] y_pred shape:", y_pred.shape)
    print("[Test 1] Nombre de noeuds dans le DAG:", len(dag.nodes))
    best = dag.get_best_node()
    if best is not None:
        print("[Test 1] Best node step:", best.step, "score:", best.score)

    # --- Test 2 : petit batch et petites dimensions ---
    x_dim2, y_dim2, z_dim2 = 16, 8, 16
    model2 = TRLinkosTRM(x_dim2, y_dim2, z_dim2)

    x_batch2 = np.random.randn(2, x_dim2)
    target2 = np.random.randn(2, y_dim2)

    def scorer2(x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.mean((y_pred - target2) ** 2, axis=-1)

    y_pred2, dag2 = model2.forward_recursive(
        x_batch2,
        max_steps=5,
        inner_recursions=2,
        scorer=scorer2,
        backtrack=False,
    )

    print("[Test 2] y_pred2 shape:", y_pred2.shape)
    print("[Test 2] Nombre de noeuds dans le DAG:", len(dag2.nodes))
    best2 = dag2.get_best_node()
    if best2 is not None:
        print("[Test 2] Best2 node step:", best2.step, "score:", best2.score)

    # --- Test 3 : Test du backtracking ---
    print("\n--- Test 3 : Backtracking ---")
    x_dim3, y_dim3, z_dim3 = 16, 8, 16
    model3 = TRLinkosTRM(x_dim3, y_dim3, z_dim3)

    x_batch3 = np.random.randn(4, x_dim3)
    target3 = np.random.randn(4, y_dim3)

    def scorer3(x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.mean((y_pred - target3) ** 2, axis=-1)

    # Sans backtracking
    y_pred3_no_bt, dag3_no_bt = model3.forward_recursive(
        x_batch3,
        max_steps=8,
        inner_recursions=2,
        scorer=scorer3,
        backtrack=False,
    )
    score_no_bt = scorer3(x_batch3, y_pred3_no_bt)
    print("[Test 3] Score sans backtracking:", np.mean(score_no_bt))

    # Avec backtracking
    np.random.seed(42)  # Reset seed for fair comparison
    model3_bt = TRLinkosTRM(x_dim3, y_dim3, z_dim3)
    y_pred3_bt, dag3_bt = model3_bt.forward_recursive(
        x_batch3,
        max_steps=8,
        inner_recursions=2,
        scorer=scorer3,
        backtrack=True,
        backtrack_threshold=0.05,
    )
    score_bt = scorer3(x_batch3, y_pred3_bt)
    print("[Test 3] Score avec backtracking:", np.mean(score_bt))
    print("[Test 3] États stockés dans DAG:", dag3_bt.store_states)

    best3 = dag3_bt.get_best_node()
    if best3 is not None:
        print("[Test 3] Best node step:", best3.step, "score:", best3.score)
        print("[Test 3] États y/z restaurables:", best3.y_state is not None)

    # --- Test 4 : Test de la structure fractale ---
    print("\n--- Test 4 : Structure Fractale du DAG ---")
    fractal_dag = FractalMerkleDAG(store_states=True, max_depth=3)

    # Créer des états simulés
    y_test = np.random.randn(1, 8)
    z_test = np.random.randn(1, 16)

    # Ajouter des noeuds au niveau racine (depth=0)
    root_id = fractal_dag.add_step(
        step=0, y=y_test, z=z_test, parents=[], score=-1.0, depth=0
    )
    node1_id = fractal_dag.add_step(
        step=1, y=y_test * 0.9, z=z_test * 0.9, parents=[root_id], score=-0.8, depth=0
    )
    node2_id = fractal_dag.add_step(
        step=2, y=y_test * 0.8, z=z_test * 0.8, parents=[node1_id], score=-0.6, depth=0
    )

    # Créer une branche fractale (depth=1)
    branch1_id = fractal_dag.create_branch(
        parent_node_id=node1_id,
        y=y_test * 1.1,
        z=z_test * 1.1,
        score=-0.7
    )

    # Ajouter des noeuds à la branche
    if branch1_id:
        branch1_node1 = fractal_dag.add_step(
            step=1, y=y_test * 1.2, z=z_test * 1.2,
            parents=[branch1_id], score=-0.5, depth=1, branch_root=node1_id
        )

        # Créer une sous-branche (depth=2)
        sub_branch_id = fractal_dag.create_branch(
            parent_node_id=branch1_node1,
            y=y_test * 1.3,
            z=z_test * 1.3,
            score=-0.4
        )

    # Afficher les statistiques fractales
    depth_stats = fractal_dag.get_depth_statistics()
    print("[Test 4] Statistiques de profondeur fractale:")
    for depth, count in sorted(depth_stats.items()):
        print(f"  Profondeur {depth}: {count} noeud(s)")

    print(f"[Test 4] Nombre total de noeuds: {len(fractal_dag.nodes)}")
    print(f"[Test 4] Profondeur maximale configurée: {fractal_dag.max_depth}")

    # Vérifier le meilleur noeud
    best_fractal = fractal_dag.get_best_node()
    if best_fractal:
        print(f"[Test 4] Meilleur noeud - depth: {best_fractal.depth}, "
              f"step: {best_fractal.step}, score: {best_fractal.score}")

    # Tester le chemin fractal
    if sub_branch_id:
        path = fractal_dag.get_fractal_path(sub_branch_id)
        print(f"[Test 4] Chemin fractal du noeud depth=2: "
              f"{len(path)} noeuds traversés")
        print(f"[Test 4] Profondeurs du chemin: "
              f"{[n.depth for n in path]}")

    # Vérifier les liens parent-enfant
    root_node = fractal_dag.nodes[root_id]
    print(f"[Test 4] Noeud racine a {len(root_node.children)} enfant(s)")

    # --- Test 5 : forward_recursive_fractal avec exploration fractale ---
    print("\n--- Test 5 : Forward Recursive avec Exploration Fractale ---")
    np.random.seed(42)
    x_dim5, y_dim5, z_dim5 = 16, 8, 16
    model5 = TRLinkosTRM(x_dim5, y_dim5, z_dim5, hidden_dim=64, num_experts=2)

    x_batch5 = np.random.randn(4, x_dim5)
    target5 = np.random.randn(4, y_dim5)

    def scorer5(x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.mean((y_pred - target5) ** 2, axis=-1)

    # Test avec exploration fractale
    y_pred5, dag5 = model5.forward_recursive_fractal(
        x_batch5,
        max_steps=6,
        inner_recursions=2,
        scorer=scorer5,
        backtrack=True,
        fractal_branching=True,
        branch_threshold=0.01,
        max_branches_per_node=2,
    )

    print(f"[Test 5] y_pred5 shape: {y_pred5.shape}")
    print(f"[Test 5] Nombre de noeuds dans le DAG: {len(dag5.nodes)}")

    # Statistiques de profondeur fractale
    depth_stats5 = dag5.get_depth_statistics()
    print("[Test 5] Statistiques de profondeur fractale:")
    for depth, count in sorted(depth_stats5.items()):
        print(f"  Profondeur {depth}: {count} noeud(s)")

    # Vérifier la structure fractale générée
    if len(depth_stats5) > 1:
        print("[Test 5] ✅ Branches fractales créées avec succès!")
    else:
        print("[Test 5] Note: Pas de branches fractales (variabilité insuffisante)")

    # --- Test 6 : TextEncoder ---
    print("\n--- Test 6 : TextEncoder ---")
    text_encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16, mode="char")

    texts = [
        "Hello world!",
        "Ceci est un test.",
        "T-RLINKOS TRM++",
        "Raisonnement récursif"
    ]

    text_embeddings = text_encoder.encode(texts, max_length=64)
    print(f"[Test 6] Texts encodés: {len(texts)}")
    print(f"[Test 6] Shape des embeddings: {text_embeddings.shape}")
    assert text_embeddings.shape == (4, 16), f"Shape incorrecte: {text_embeddings.shape}"
    print("[Test 6] ✅ TextEncoder fonctionne correctement!")

    # Test mode word
    word_encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16, mode="word")
    word_embeddings = word_encoder.encode(texts, max_length=32)
    print(f"[Test 6] Word embeddings shape: {word_embeddings.shape}")
    assert word_embeddings.shape == (4, 16), f"Shape incorrecte: {word_embeddings.shape}"

    # --- Test 7 : ImageEncoder ---
    print("\n--- Test 7 : ImageEncoder ---")
    image_encoder = ImageEncoder(input_channels=3, patch_size=4, embed_dim=32, output_dim=16)

    # Créer des images factices (RGB 32x32)
    images = [np.random.rand(32, 32, 3) for _ in range(4)]

    image_embeddings = image_encoder.encode(images)
    print(f"[Test 7] Images encodées: {len(images)}")
    print(f"[Test 7] Shape des embeddings: {image_embeddings.shape}")
    assert image_embeddings.shape == (4, 16), f"Shape incorrecte: {image_embeddings.shape}"

    # Test avec images grayscale
    gray_images = [np.random.rand(32, 32) for _ in range(2)]
    gray_encoder = ImageEncoder(input_channels=1, patch_size=4, embed_dim=32, output_dim=16)
    gray_embeddings = gray_encoder.encode(gray_images)
    print(f"[Test 7] Grayscale embeddings shape: {gray_embeddings.shape}")
    print("[Test 7] ✅ ImageEncoder fonctionne correctement!")

    # --- Test 8 : Dataset et DataLoader ---
    print("\n--- Test 8 : Dataset et DataLoader ---")
    x_dim8, y_dim8 = 16, 8

    # Dataset vectoriel simple
    dataset = Dataset(x_dim=x_dim8, y_dim=y_dim8, encoder_type="vector")
    for i in range(20):
        x = np.random.randn(x_dim8)
        y = np.random.randn(y_dim8)
        dataset.add_sample(x, y, metadata={"index": i})

    print(f"[Test 8] Taille du dataset: {len(dataset)}")
    assert len(dataset) == 20

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    num_batches = len(dataloader)
    print(f"[Test 8] Nombre de batches: {num_batches}")

    # Itérer sur les batches
    total_samples = 0
    for x_batch, y_batch in dataloader:
        total_samples += x_batch.shape[0]
        assert x_batch.shape[1] == x_dim8
        assert y_batch.shape[1] == y_dim8

    print(f"[Test 8] Total échantillons itérés: {total_samples}")
    print("[Test 8] ✅ Dataset et DataLoader fonctionnent correctement!")

    # --- Test 9 : Loss Functions ---
    print("\n--- Test 9 : Loss Functions ---")
    y_pred9 = np.random.randn(8, 10)
    y_target9 = np.random.randn(8, 10)

    mse = mse_loss(y_pred9, y_target9)
    print(f"[Test 9] MSE Loss: {mse:.6f}")
    assert mse >= 0, "MSE doit être non-négative"

    # Cross-entropy avec indices de classe
    logits = np.random.randn(8, 5)
    class_indices = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    ce_loss = cross_entropy_loss(logits, class_indices)
    print(f"[Test 9] Cross-Entropy Loss: {ce_loss:.6f}")
    assert ce_loss >= 0, "Cross-entropy doit être non-négative"

    # Cosine similarity loss
    cos_loss = cosine_similarity_loss(y_pred9, y_target9)
    print(f"[Test 9] Cosine Similarity Loss: {cos_loss:.6f}")
    assert 0 <= cos_loss <= 2, "Cosine loss doit être entre 0 et 2"

    print("[Test 9] ✅ Loss functions fonctionnent correctement!")

    # --- Test 10 : Training Pipeline avec données synthétiques ---
    print("\n--- Test 10 : Training Pipeline ---")
    np.random.seed(42)

    # Petit modèle pour test rapide
    x_dim10, y_dim10, z_dim10 = 8, 4, 8
    model10 = TRLinkosTRM(x_dim10, y_dim10, z_dim10, hidden_dim=32, num_experts=2)

    # Dataset d'entraînement synthétique
    train_dataset = Dataset(x_dim=x_dim10, y_dim=y_dim10)
    for i in range(16):
        x = np.random.randn(x_dim10)
        # Cible = transformation simple de x pour avoir un pattern appris
        y = np.tanh(x[:y_dim10] + 0.5)
        train_dataset.add_sample(x, y)

    # Configuration d'entraînement courte
    config = TrainingConfig(
        learning_rate=0.001,
        num_epochs=3,  # Très peu d'époques pour le test
        batch_size=4,
        max_steps=4,
        inner_recursions=2,
        use_fractal_branching=False,
        loss_fn="mse",
        log_interval=1,
        gradient_clip=1.0
    )

    # Trainer
    trainer = Trainer(model10, config)

    # Entraînement
    history = trainer.train(train_dataset)

    print(f"[Test 10] Nombre d'époques: {len(history['train_loss'])}")
    assert len(history["train_loss"]) == 3, "Historique incorrect"

    # Évaluation
    eval_loss, predictions = trainer.evaluate(train_dataset)
    print(f"[Test 10] Eval Loss: {eval_loss:.6f}")
    print(f"[Test 10] Predictions shape: {predictions.shape}")

    print("[Test 10] ✅ Training Pipeline fonctionne correctement!")

    # --- Test 11 : Training avec TextEncoder ---
    print("\n--- Test 11 : Training avec Données Textuelles ---")
    np.random.seed(42)

    text_enc = TextEncoder(vocab_size=256, embed_dim=16, output_dim=8, mode="char")
    text_dataset = Dataset(
        x_dim=8, y_dim=4,
        encoder_type="text",
        text_encoder=text_enc
    )

    # Ajouter des échantillons textuels
    sample_texts = [
        ("Hello", [0.5, 0.5, 0.5, 0.5]),
        ("World", [-0.5, -0.5, -0.5, -0.5]),
        ("Test", [0.1, 0.1, 0.1, 0.1]),
        ("AI", [0.8, 0.8, 0.8, 0.8]),
    ]
    for text, target in sample_texts:
        text_dataset.add_sample(text, np.array(target))

    print(f"[Test 11] Dataset textuel: {len(text_dataset)} échantillons")

    # Vérifier que les données sont bien encodées
    sample = text_dataset[0]
    print(f"[Test 11] Premier échantillon x shape: {sample.x.shape}")
    assert sample.x.shape == (8,), f"Shape incorrecte: {sample.x.shape}"
    print("[Test 11] ✅ Dataset textuel fonctionne correctement!")

    # --- Test 12 : Training avec ImageEncoder ---
    print("\n--- Test 12 : Training avec Données Image ---")
    np.random.seed(42)

    img_enc = ImageEncoder(input_channels=3, patch_size=4, embed_dim=16, output_dim=8)
    image_dataset = Dataset(
        x_dim=8, y_dim=4,
        encoder_type="image",
        image_encoder=img_enc
    )

    # Ajouter des images factices
    for i in range(4):
        img = np.random.rand(16, 16, 3)  # Petite image RGB
        target = np.random.randn(4)
        image_dataset.add_sample(img, target)

    print(f"[Test 12] Dataset image: {len(image_dataset)} échantillons")

    # Vérifier que les données sont bien encodées
    sample = image_dataset[0]
    print(f"[Test 12] Premier échantillon x shape: {sample.x.shape}")
    assert sample.x.shape == (8,), f"Shape incorrecte: {sample.x.shape}"
    print("[Test 12] ✅ Dataset image fonctionne correctement!")

    # --- Test 13 : Model Serialization ---
    print("\n--- Test 13 : Model Serialization ---")
    import tempfile
    import os

    np.random.seed(42)
    x_dim13, y_dim13, z_dim13 = 16, 8, 16
    model13 = TRLinkosTRM(x_dim13, y_dim13, z_dim13, hidden_dim=64, num_experts=2)

    # Create test input
    x_test = np.random.randn(4, x_dim13)

    # Get prediction before saving
    y_pred_before, _ = model13.forward_recursive(x_test, max_steps=3)

    # Save model to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.npz")
        save_model(model13, filepath)
        print(f"[Test 13] Model saved to: {filepath}")

        # Check file exists
        assert os.path.exists(filepath), "Model file not created"
        file_size = os.path.getsize(filepath)
        print(f"[Test 13] File size: {file_size / 1024:.2f} KB")

        # Load model
        model13_loaded = load_model(filepath)
        print("[Test 13] Model loaded successfully")

        # Verify dimensions match
        assert model13_loaded.x_dim == x_dim13, "x_dim mismatch"
        assert model13_loaded.y_dim == y_dim13, "y_dim mismatch"
        assert model13_loaded.z_dim == z_dim13, "z_dim mismatch"

        # Get prediction after loading
        y_pred_after, _ = model13_loaded.forward_recursive(x_test, max_steps=3)

        # Verify predictions match
        np.testing.assert_array_almost_equal(
            y_pred_before, y_pred_after, decimal=10,
            err_msg="Predictions differ after load"
        )
        print("[Test 13] Predictions match after save/load")

    print("[Test 13] ✅ Model serialization fonctionne correctement!")

    # --- Test 14 : Formal Benchmarks ---
    print("\n--- Test 14 : Formal Benchmarks ---")
    np.random.seed(42)

    # Create a small model for benchmarking
    x_dim14, y_dim14, z_dim14 = 16, 8, 16
    model14 = TRLinkosTRM(x_dim14, y_dim14, z_dim14, hidden_dim=32, num_experts=2)

    # Run single benchmark
    result = benchmark_forward_recursive(
        model14,
        batch_size=4,
        max_steps=4,
        inner_recursions=2,
        num_runs=2,
        warmup_runs=1,
    )

    print(f"[Test 14] Benchmark name: {result.name}")
    print(f"[Test 14] Total time: {result.total_time * 1000:.2f} ms")
    print(f"[Test 14] Throughput: {result.throughput:.1f} samples/sec")
    print(f"[Test 14] Memory estimate: {result.memory_estimate_mb:.2f} MB")

    # Verify result structure
    assert result.name == "forward_recursive"
    assert result.total_time > 0
    assert result.throughput > 0
    assert result.batch_size == 4
    assert result.num_steps == 4

    # Run fractal benchmark
    result_fractal = benchmark_forward_recursive_fractal(
        model14,
        batch_size=4,
        max_steps=4,
        inner_recursions=2,
        num_runs=2,
        warmup_runs=1,
    )

    print(f"[Test 14] Fractal benchmark time: {result_fractal.total_time * 1000:.2f} ms")
    assert result_fractal.name == "forward_recursive_fractal"

    # Test BenchmarkResult dataclass
    assert isinstance(result.config, dict)
    assert "x_dim" in result.config
    assert "num_experts" in result.config

    print("[Test 14] ✅ Formal benchmarks fonctionnent correctement!")

    print("\n" + "=" * 50)
    print("✅ Tous les tests passent avec succès!")
    print("=" * 50)
