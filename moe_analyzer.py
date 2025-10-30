"""
MoE Analyzer

Core analysis module for computing expert activation statistics and layer correlations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
from scipy.stats import pearsonr
from collections import defaultdict

try:
    from .moe_hooks import MoEHookManager
except ImportError:
    from moe_hooks import MoEHookManager


class MoEAnalyzer:
    """
    Main analyzer class for MoE expert activation analysis.
    
    This class provides methods to:
    - Collect routing data during inference
    - Compute expert activation statistics
    - Analyze layer-to-layer correlations
    - Detect periodic patterns (e.g., delta=12 patterns)
    - Compute router and expert weight similarities
    """
    
    def __init__(self, model, model_type: str = "auto"):
        """
        Initialize the MoE analyzer.
        
        Args:
            model: The transformer model to analyze
            model_type: One of "minimax", "deepseek_v3", "qwen3", or "auto"
        """
        self.model = model
        self.model_type = model_type
        self.hook_manager = MoEHookManager(model, model_type)
        self.activation_stats = None
        self.layer_indices = []
        
    @contextmanager
    def record(self):
        """
        Context manager for recording MoE activations during forward passes.
        
        Example:
            with analyzer.record():
                outputs = model.generate(inputs, max_length=100)
        """
        self.hook_manager.clear_data()
        num_hooks = self.hook_manager.register_hooks()
        
        try:
            yield
        finally:
            self.hook_manager.remove_hooks()
            # Process collected data
            self._process_collected_data()
    
    def _process_collected_data(self):
        """Process and aggregate collected hook data."""
        print("Processing collected data...")
        aggregated = self.hook_manager.aggregate_data()
        self.layer_indices = sorted(aggregated.keys())
        
        # Compute activation statistics for each layer
        self.activation_stats = {}
        for layer_idx, data in aggregated.items():
            routing_probs = data['routing_probs']  # [num_tokens, num_experts]
            
            # Compute average activation probability per expert
            avg_activation = routing_probs.mean(dim=0).numpy()  # [num_experts]
            
            # Compute activation variance
            var_activation = routing_probs.var(dim=0).numpy()
            
            # Get top-k info
            topk_probs = data['topk_probs']  # [num_tokens, top_k]
            selected_experts = data['selected_experts']  # [num_tokens, top_k]
            
            self.activation_stats[layer_idx] = {
                'avg_activation': avg_activation,
                'var_activation': var_activation,
                'routing_probs': routing_probs,  # Keep for correlation analysis
                'topk_probs': topk_probs,
                'selected_experts': selected_experts,
                'num_tokens': routing_probs.shape[0],
                'num_experts': routing_probs.shape[1],
            }
        
        print(f"Processed data from {len(self.activation_stats)} MoE layers")
    
    def get_expert_activation_matrix(self) -> np.ndarray:
        """
        Get the expert activation matrix (layers × experts).
        
        Returns:
            2D array of shape [num_layers, num_experts] with average activation probabilities
        """
        if not self.activation_stats:
            raise ValueError("No data collected. Use the record() context manager first.")
        
        layers = sorted(self.activation_stats.keys())
        num_experts = self.activation_stats[layers[0]]['num_experts']
        
        matrix = np.zeros((len(layers), num_experts))
        for i, layer_idx in enumerate(layers):
            matrix[i] = self.activation_stats[layer_idx]['avg_activation']
        
        return matrix
    
    def compute_layer_correlation_matrix(self, delta_max: int = 30) -> Dict[str, np.ndarray]:
        """
        Compute correlation between expert activations at different layer distances (delta).
        
        This analyzes whether the same expert index tends to be activated across layers
        separated by a fixed distance (delta).
        
        Args:
            delta_max: Maximum delta (layer distance) to analyze
            
        Returns:
            Dictionary containing:
            - 'correlation_matrix': [num_layers, delta_max] correlation values
            - 'deltas': List of delta values
            - 'layer_pairs': List of (layer_i, layer_j, delta, correlation) tuples
        """
        if not self.activation_stats:
            raise ValueError("No data collected. Use the record() context manager first.")
        
        layers = sorted(self.activation_stats.keys())
        num_layers = len(layers)
        
        # Initialize correlation matrix: rows=starting layer, cols=delta
        correlation_matrix = np.full((num_layers, delta_max), np.nan)
        layer_pairs = []
        
        for i, layer_i in enumerate(layers):
            probs_i = self.activation_stats[layer_i]['routing_probs'].numpy()  # [tokens, experts]
            
            for delta in range(1, delta_max + 1):
                if i + delta >= num_layers:
                    break
                
                layer_j_idx = i + delta
                layer_j = layers[layer_j_idx]
                probs_j = self.activation_stats[layer_j]['routing_probs'].numpy()
                
                # Compute correlation between activation probabilities
                # Average across tokens, then correlate expert activation profiles
                avg_probs_i = probs_i.mean(axis=0)  # [num_experts]
                avg_probs_j = probs_j.mean(axis=0)  # [num_experts]
                
                if len(avg_probs_i) == len(avg_probs_j):
                    corr, _ = pearsonr(avg_probs_i, avg_probs_j)
                    correlation_matrix[i, delta - 1] = corr
                    layer_pairs.append((layer_i, layer_j, delta, corr))
        
        return {
            'correlation_matrix': correlation_matrix,
            'deltas': list(range(1, delta_max + 1)),
            'layer_pairs': layer_pairs,
            'layer_indices': layers,
        }
    
    def compute_periodic_patterns(self, intervals: List[int] = [12, 24]) -> Dict[int, Dict[str, float]]:
        """
        Analyze periodic patterns at specific intervals (e.g., delta=12, delta=24).
        
        Args:
            intervals: List of deltas to specifically analyze
            
        Returns:
            Dictionary mapping intervals to statistics:
            - 'mean_correlation': Average correlation at this delta
            - 'std_correlation': Standard deviation
            - 'num_pairs': Number of layer pairs with this delta
        """
        if not self.activation_stats:
            raise ValueError("No data collected. Use the record() context manager first.")
        
        layers = sorted(self.activation_stats.keys())
        results = {}
        
        for interval in intervals:
            correlations = []
            
            for i, layer_i in enumerate(layers):
                if i + interval >= len(layers):
                    break
                
                layer_j = layers[i + interval]
                
                probs_i = self.activation_stats[layer_i]['routing_probs'].numpy()
                probs_j = self.activation_stats[layer_j]['routing_probs'].numpy()
                
                avg_probs_i = probs_i.mean(axis=0)
                avg_probs_j = probs_j.mean(axis=0)
                
                if len(avg_probs_i) == len(avg_probs_j):
                    corr, _ = pearsonr(avg_probs_i, avg_probs_j)
                    correlations.append(corr)
            
            if correlations:
                results[interval] = {
                    'mean_correlation': np.mean(correlations),
                    'std_correlation': np.std(correlations),
                    'num_pairs': len(correlations),
                    'all_correlations': correlations,
                }
        
        return results
    
    def compute_router_weight_similarity(self) -> Dict[str, np.ndarray]:
        """
        Compute similarity between router weights across layers.
        
        Returns:
            Dictionary containing:
            - 'cosine_similarity_matrix': [num_layers, num_layers] pairwise cosine similarities
            - 'column_norm_correlation': Correlation of column norms between routers
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            raise ValueError("Model structure not recognized")
        
        layers = sorted(self.activation_stats.keys()) if self.activation_stats else []
        if not layers:
            # Try to get all layers
            layers = list(range(len(self.model.model.layers)))
        
        router_weights = []
        router_layers = []
        
        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx]
            
            # Get router/gate weights
            moe_module = None
            if hasattr(layer, 'block_sparse_moe'):
                moe_module = layer.block_sparse_moe
            elif hasattr(layer, 'mlp'):
                mlp_class_name = layer.mlp.__class__.__name__
                if any(x in mlp_class_name for x in ["SparseMoe", "MoE", "Moe"]):
                    moe_module = layer.mlp
            
            if moe_module and hasattr(moe_module, 'gate'):
                gate = moe_module.gate
                if hasattr(gate, 'weight'):
                    weight = gate.weight.detach().cpu().float().numpy()
                    router_weights.append(weight)
                    router_layers.append(layer_idx)
        
        if len(router_weights) < 2:
            return {'error': 'Not enough router weights found'}
        
        # Compute pairwise cosine similarities
        num_routers = len(router_weights)
        cosine_sim_matrix = np.zeros((num_routers, num_routers))
        
        for i in range(num_routers):
            for j in range(num_routers):
                w_i = router_weights[i].flatten()
                w_j = router_weights[j].flatten()
                
                # Cosine similarity
                cos_sim = np.dot(w_i, w_j) / (np.linalg.norm(w_i) * np.linalg.norm(w_j) + 1e-8)
                cosine_sim_matrix[i, j] = cos_sim
        
        # Compute column norm correlation for each delta
        column_norms = [np.linalg.norm(w, axis=1) for w in router_weights]  # [num_experts] per router
        
        delta_max = min(30, num_routers - 1)
        norm_correlations = np.full((num_routers, delta_max), np.nan)
        
        for i in range(num_routers):
            for delta in range(1, delta_max + 1):
                if i + delta < num_routers:
                    corr, _ = pearsonr(column_norms[i], column_norms[i + delta])
                    norm_correlations[i, delta - 1] = corr
        
        return {
            'cosine_similarity_matrix': cosine_sim_matrix,
            'column_norm_correlation': norm_correlations,
            'router_layers': router_layers,
        }
    
    def compute_expert_weight_similarity(self, delta: int = 12) -> Dict[str, Any]:
        """
        Compute similarity between expert weights at layers separated by delta.
        
        Compares the up_proj and down_proj (or equivalent) weights of experts
        at the same expert index but different layers.
        
        Args:
            delta: Layer distance to compare
            
        Returns:
            Dictionary with similarity statistics and distributions
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            raise ValueError("Model structure not recognized")
        
        layers = sorted(self.activation_stats.keys()) if self.activation_stats else []
        if not layers:
            layers = list(range(len(self.model.model.layers)))
        
        similarities = []
        
        for i, layer_i in enumerate(layers):
            if i + delta >= len(layers):
                break
            
            layer_j = layers[i + delta]
            
            # Get expert modules
            experts_i = self._get_experts_from_layer(self.model.model.layers[layer_i])
            experts_j = self._get_experts_from_layer(self.model.model.layers[layer_j])
            
            if experts_i is None or experts_j is None:
                continue
            
            num_experts = min(len(experts_i), len(experts_j))
            
            for expert_idx in range(num_experts):
                expert_sim = self._compute_expert_pair_similarity(
                    experts_i[expert_idx],
                    experts_j[expert_idx]
                )
                if expert_sim is not None:
                    similarities.append({
                        'layer_i': layer_i,
                        'layer_j': layer_j,
                        'expert_idx': expert_idx,
                        'similarity': expert_sim,
                    })
        
        if not similarities:
            return {'error': 'No expert weights found for comparison'}
        
        similarity_values = [s['similarity'] for s in similarities]
        
        return {
            'similarities': similarities,
            'mean_similarity': np.mean(similarity_values),
            'std_similarity': np.std(similarity_values),
            'median_similarity': np.median(similarity_values),
            'delta': delta,
        }
    
    def _get_experts_from_layer(self, layer):
        """Extract expert modules from a layer."""
        if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
            return layer.block_sparse_moe.experts
        elif hasattr(layer, 'mlp'):
            if hasattr(layer.mlp, 'experts'):
                return layer.mlp.experts
        return None
    
    def _compute_expert_pair_similarity(self, expert1, expert2) -> Optional[float]:
        """Compute cosine similarity between two expert modules."""
        # Get weights from up_proj and down_proj (or equivalent)
        weights1 = []
        weights2 = []
        
        for name in ['up_proj', 'down_proj', 'gate_proj', 'w1', 'w2', 'w3']:
            if hasattr(expert1, name) and hasattr(expert2, name):
                w1 = getattr(expert1, name).weight.detach().cpu().float().numpy().flatten()
                w2 = getattr(expert2, name).weight.detach().cpu().float().numpy().flatten()
                weights1.append(w1)
                weights2.append(w2)
        
        if not weights1:
            return None
        
        # Concatenate all weights
        w1_all = np.concatenate(weights1)
        w2_all = np.concatenate(weights2)
        
        # Compute cosine similarity
        cos_sim = np.dot(w1_all, w2_all) / (np.linalg.norm(w1_all) * np.linalg.norm(w2_all) + 1e-8)
        
        return float(cos_sim)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the analysis."""
        if not self.activation_stats:
            return {'error': 'No data collected'}
        
        layers = sorted(self.activation_stats.keys())
        
        summary = {
            'num_moe_layers': len(layers),
            'layer_indices': layers,
            'num_experts_per_layer': {},
            'top_k_per_layer': {},
            'total_tokens_analyzed': self.activation_stats[layers[0]]['num_tokens'],
        }
        
        for layer_idx in layers:
            stats = self.activation_stats[layer_idx]
            summary['num_experts_per_layer'][layer_idx] = stats['num_experts']
            summary['top_k_per_layer'][layer_idx] = stats['topk_probs'].shape[1]
        
        return summary

