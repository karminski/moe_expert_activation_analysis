"""
MoE Hook Manager

Captures router outputs and expert activations from MoE layers during forward passes.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


class MoEHookManager:
    """
    Manages hooks for capturing MoE layer outputs across different model architectures.
    
    Supported models:
    - MiniMax: MiniMaxSparseMoeBlock
    - DeepSeek-V3: DeepseekV3MoE
    - Qwen3: Qwen3MoeSparseMoeBlock
    """
    
    def __init__(self, model, model_type: str = "auto"):
        """
        Initialize the hook manager.
        
        Args:
            model: The transformer model to analyze
            model_type: One of "minimax", "deepseek_v3", "qwen3", or "auto" for automatic detection
        """
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type == "auto" else model_type
        self.hooks = []
        self.collected_data = defaultdict(list)
        self.layer_to_idx = {}  # Maps layer objects to their indices
        
    def _detect_model_type(self, model) -> str:
        """Automatically detect the model type based on architecture."""
        model_class_name = model.__class__.__name__
        
        if "MiniMax" in model_class_name:
            return "minimax"
        elif "DeepseekV3" in model_class_name or "Deepseek" in model_class_name:
            return "deepseek_v3"
        elif "Qwen3Moe" in model_class_name or "Qwen3" in model_class_name:
            return "qwen3"
        else:
            # Try to infer from the first MoE layer found
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for layer in model.model.layers:
                    if hasattr(layer, 'mlp') or hasattr(layer, 'block_sparse_moe'):
                        mlp_name = layer.mlp.__class__.__name__ if hasattr(layer, 'mlp') else layer.block_sparse_moe.__class__.__name__
                        if "MiniMax" in mlp_name or "Mixtral" in mlp_name:
                            return "minimax"
                        elif "Deepseek" in mlp_name:
                            return "deepseek_v3"
                        elif "Qwen" in mlp_name:
                            return "qwen3"
            
            raise ValueError(f"Unable to detect model type for {model_class_name}. Please specify manually.")
    
    def _is_moe_layer(self, layer) -> bool:
        """Check if a layer contains MoE components."""
        if hasattr(layer, 'mlp'):
            mlp_class_name = layer.mlp.__class__.__name__
            moe_indicators = ["SparseMoe", "MoE", "Moe"]
            return any(indicator in mlp_class_name for indicator in moe_indicators)
        elif hasattr(layer, 'block_sparse_moe'):
            return True
        return False
    
    def _create_hook_fn(self, layer_idx: int, module_name: str):
        """
        Create a hook function that captures MoE outputs.
        
        The hook captures different information based on model type:
        - MiniMax/Qwen3: Returns (hidden_states, router_logits)
        - DeepSeek-V3: MoE module returns hidden_states, but we hook the router separately
        """
        def hook_fn(module, input, output):
            if self.model_type in ["minimax", "qwen3"]:
                # Output is a tuple: (hidden_states, router_logits)
                if isinstance(output, tuple) and len(output) >= 2:
                    hidden_states, router_logits = output[0], output[1]
                    
                    # Compute routing probabilities from logits
                    routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                    
                    # Get top-k selections
                    top_k = getattr(module, 'top_k', 2)
                    topk_probs, selected_experts = torch.topk(routing_probs, top_k, dim=-1)
                    
                    self.collected_data[layer_idx].append({
                        'router_logits': router_logits.detach().cpu(),
                        'routing_probs': routing_probs.detach().cpu(),
                        'topk_probs': topk_probs.detach().cpu(),
                        'selected_experts': selected_experts.detach().cpu(),
                    })
            
            elif self.model_type == "deepseek_v3":
                # For DeepSeek-V3, we need to hook the gate (router) separately
                # The output is just hidden_states
                # We'll capture router info in a separate hook on the gate module
                pass
        
        return hook_fn
    
    def _create_deepseek_router_hook(self, layer_idx: int):
        """Create a hook specifically for DeepSeek-V3 router."""
        def hook_fn(module, input, output):
            # Output is (topk_indices, topk_weights)
            topk_indices, topk_weights = output
            
            # Reconstruct full routing probabilities
            batch_tokens = topk_indices.shape[0]
            num_experts = module.n_routed_experts
            
            # Create full probability matrix
            routing_probs = torch.zeros(batch_tokens, num_experts, device=topk_indices.device, dtype=topk_weights.dtype)
            routing_probs.scatter_(1, topk_indices, topk_weights)
            
            self.collected_data[layer_idx].append({
                'router_logits': None,  # DeepSeek uses sigmoid, not softmax
                'routing_probs': routing_probs.detach().cpu(),
                'topk_probs': topk_weights.detach().cpu(),
                'selected_experts': topk_indices.detach().cpu(),
            })
        
        return hook_fn
    
    def register_hooks(self) -> int:
        """
        Register hooks on all MoE layers in the model.
        
        Returns:
            Number of MoE layers found and hooked
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            raise ValueError("Model structure not recognized. Expected model.model.layers")
        
        moe_count = 0
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            if self._is_moe_layer(layer):
                # Determine which module to hook
                if hasattr(layer, 'block_sparse_moe'):
                    moe_module = layer.block_sparse_moe
                elif hasattr(layer, 'mlp'):
                    moe_module = layer.mlp
                else:
                    continue
                
                self.layer_to_idx[id(moe_module)] = layer_idx
                
                if self.model_type == "deepseek_v3":
                    # Hook the router (gate) for DeepSeek-V3
                    if hasattr(moe_module, 'gate'):
                        hook = moe_module.gate.register_forward_hook(
                            self._create_deepseek_router_hook(layer_idx)
                        )
                        self.hooks.append(hook)
                        moe_count += 1
                else:
                    # Hook the MoE module directly for MiniMax/Qwen3
                    hook = moe_module.register_forward_hook(
                        self._create_hook_fn(layer_idx, f"layer_{layer_idx}_moe")
                    )
                    self.hooks.append(hook)
                    moe_count += 1
        
        print(f"Registered hooks on {moe_count} MoE layers (model_type: {self.model_type})")
        return moe_count
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print(f"Removed all hooks")
    
    def clear_data(self):
        """Clear collected data."""
        self.collected_data.clear()
    
    def get_collected_data(self) -> Dict[int, List[Dict[str, torch.Tensor]]]:
        """
        Get the collected data from all hooks.
        
        Returns:
            Dictionary mapping layer indices to lists of captured data
        """
        return dict(self.collected_data)
    
    def aggregate_data(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Aggregate collected data across multiple forward passes.
        
        Returns:
            Dictionary with aggregated statistics per layer
        """
        aggregated = {}
        
        for layer_idx, data_list in self.collected_data.items():
            if not data_list:
                continue
            
            # Concatenate all data from this layer
            router_logits_list = [d['router_logits'] for d in data_list if d['router_logits'] is not None]
            routing_probs_list = [d['routing_probs'] for d in data_list]
            topk_probs_list = [d['topk_probs'] for d in data_list]
            selected_experts_list = [d['selected_experts'] for d in data_list]
            
            aggregated[layer_idx] = {
                'router_logits': torch.cat(router_logits_list, dim=0) if router_logits_list else None,
                'routing_probs': torch.cat(routing_probs_list, dim=0),
                'topk_probs': torch.cat(topk_probs_list, dim=0),
                'selected_experts': torch.cat(selected_experts_list, dim=0),
            }
        
        return aggregated

