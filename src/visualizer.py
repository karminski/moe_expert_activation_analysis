"""
MoE Visualizer

Creates interactive visualizations using Plotly for MoE analysis results.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import os
import json
import pickle


class MoEVisualizer:
    """
    Creates interactive visualizations for MoE analysis results.

    All plots are generated as HTML files that can be opened in a browser.
    """

    def __init__(self):
        """Initialize the visualizer."""
        pass

    def plot_expert_activation_3d(
        self,
        analyzer,
        save_path: str = "expert_activation_3d.html",
        max_tokens: int = 50,
        max_layers: int = 10,
        max_experts: int = 64,
    ):
        """
        Create a 3D surface visualization of expert activation probabilities.
        
        Args:
            analyzer: MoEAnalyzer instance with collected data
            save_path: Path to save the HTML file
            max_tokens: Maximum number of tokens to visualize (for performance)
            max_layers: Maximum number of layers to visualize
            max_experts: Maximum number of experts to visualize
        """
        print(f"\nGenerating 3D expert activation visualization...")
        
        if not hasattr(analyzer, 'activation_stats') or not analyzer.activation_stats:
            print("⚠️  No activation data available for 3D visualization")
            return
        
        # Select layers to visualize
        all_layers = sorted(analyzer.activation_stats.keys())
        layer_step = max(1, len(all_layers) // max_layers)
        selected_layers = all_layers[::layer_step][:max_layers]
        
        # Create figure
        fig = go.Figure()
        
        # Prepare data for each layer
        for layer_idx, layer in enumerate(selected_layers):
            stats = analyzer.activation_stats[layer]
            routing_probs = stats['routing_probs'].numpy()  # [num_tokens, num_experts]
            
            # Limit tokens and experts for visualization
            num_tokens = min(max_tokens, routing_probs.shape[0])
            num_experts = min(max_experts, routing_probs.shape[1])
            routing_probs = routing_probs[:num_tokens, :num_experts]
            
            # Transpose to have experts on X-axis and tokens on Y-axis
            # This matches the typical layout where experts are columns
            z_data = routing_probs.T  # [num_experts, num_tokens]
            
            # Create meshgrids
            expert_indices = np.arange(num_experts)
            token_indices = np.arange(num_tokens)
            
            # Add surface plot for this layer
            # Offset layers in the Z direction to create stacked effect
            z_offset = layer_idx * 0.3  # Stack layers
            
            # Create surface with custom colorscale
            # Custom gradient from deep blue (#19448e) to pink (#f4b3c2)
            custom_colorscale = [
                [0.0, '#19448e'],   # Deep blue (minimum)
                [0.25, '#4a6fa5'],  # Medium blue
                [0.5, '#7d8db8'],   # Light blue-purple
                [0.75, '#c49fbb'],  # Light purple-pink
                [1.0, '#f4b3c2']    # Pink (maximum)
            ]
            
            fig.add_trace(
                go.Surface(
                    x=expert_indices,
                    y=token_indices,
                    z=z_data + z_offset,  # Add offset for each layer
                    colorscale=custom_colorscale,
                    opacity=0.85,
                    name=f"Layer {layer}",
                    hovertemplate="Expert: %{x}<br>Token: %{y}<br>Activation: %{z:.4f}<br>Layer: " + str(layer) + "<extra></extra>",
                    showscale=(layer_idx == 0),  # Only show colorbar for first layer
                    colorbar=dict(
                        title="Activation<br>Probability",
                        x=1.05,
                        len=0.7,
                    ) if layer_idx == 0 else None,
                    cmin=0,
                    cmax=1,
                )
            )
        
        # Update layout with better camera angle
        fig.update_layout(
            title=dict(
                text="3D Expert Activation Visualization<br><sub>Surface height represents activation probability across layers</sub>",
                x=0.5,
                xanchor='center',
            ),
            scene=dict(
                xaxis=dict(
                    title="Expert Index",
                    gridcolor="lightgray",
                    backgroundcolor="rgba(230, 230, 250, 0.5)",
                    showbackground=True,
                ),
                yaxis=dict(
                    title="Token Position",
                    gridcolor="lightgray",
                    backgroundcolor="rgba(230, 250, 230, 0.5)",
                    showbackground=True,
                ),
                zaxis=dict(
                    title="Activation Probability (+ Layer Offset)",
                    gridcolor="lightgray",
                    backgroundcolor="rgba(250, 230, 230, 0.5)",
                    showbackground=True,
                ),
                camera=dict(
                    eye=dict(x=1.8, y=-1.8, z=1.5),  # Isometric-like view
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.8),
            ),
            width=1400,
            height=1000,
            hovermode='closest',
            paper_bgcolor='rgba(240, 240, 250, 0.95)',
            font=dict(size=12),
        )
        
        fig.write_html(save_path)
        print(f"✓ Saved 3D visualization to {save_path}")
        print(f"   Visualized {len(selected_layers)} layers with {num_experts} experts and {num_tokens} tokens")

    def plot_expert_activation_heatmap(
        self,
        activation_matrix: np.ndarray,
        layer_indices: Optional[List[int]] = None,
        save_path: str = "expert_activation_heatmap.html",
        title: str = "Expert Activation Probability Heatmap",
    ):
        """
        Create a heatmap of expert activation probabilities.

        Args:
            activation_matrix: [num_layers, num_experts] array of activation probabilities
            layer_indices: List of layer indices (for y-axis labels)
            save_path: Path to save the HTML file
            title: Plot title
        """
        num_layers, num_experts = activation_matrix.shape

        if layer_indices is None:
            layer_indices = list(range(num_layers))

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=activation_matrix,
                x=[f"Expert {i}" for i in range(num_experts)],
                y=[f"Layer {idx}" for idx in layer_indices],
                colorscale="Viridis",
                colorbar=dict(title="Avg Activation<br>Probability"),
                hovertemplate="Layer: %{y}<br>Expert: %{x}<br>Activation: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Expert Index",
            yaxis_title="Layer Index",
            width=max(800, num_experts * 15),
            height=max(600, num_layers * 20),
            font=dict(size=12),
        )

        # Save to file
        fig.write_html(save_path)
        print(f"Saved expert activation heatmap to {save_path}")

    def plot_layer_correlation_matrix(
        self,
        correlation_data: Dict[str, Any],
        save_path: str = "layer_correlation_matrix.html",
        title: str = "Layer-to-Layer Correlation Matrix",
    ):
        """
        Create a heatmap showing correlation between layers at different deltas.

        Args:
            correlation_data: Dictionary from compute_layer_correlation_matrix()
            save_path: Path to save the HTML file
            title: Plot title
        """
        correlation_matrix = correlation_data["correlation_matrix"]
        layer_indices = correlation_data["layer_indices"]
        deltas = correlation_data["deltas"]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix,
                x=[f"Δ={d}" for d in deltas],
                y=[f"Layer {idx}" for idx in layer_indices],
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Pearson<br>Correlation"),
                hovertemplate="Start Layer: %{y}<br>Delta: %{x}<br>Correlation: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Layer Distance (Δ)",
            yaxis_title="Starting Layer",
            width=max(1000, len(deltas) * 20),
            height=max(600, len(layer_indices) * 15),
            font=dict(size=12),
        )

        # Save to file
        fig.write_html(save_path)
        print(f"Saved layer correlation matrix to {save_path}")

    def plot_periodic_pattern(
        self,
        periodic_data: Dict[int, Dict[str, Any]],
        save_path: str = "periodic_pattern_analysis.html",
        title: str = "Periodic Pattern Analysis",
    ):
        """
        Visualize periodic patterns at specific intervals.

        Args:
            periodic_data: Dictionary from compute_periodic_patterns()
            save_path: Path to save the HTML file
            title: Plot title
        """
        intervals = sorted(periodic_data.keys())

        # Create bar chart for mean correlations
        mean_corrs = [periodic_data[i]["mean_correlation"] for i in intervals]
        std_corrs = [periodic_data[i]["std_correlation"] for i in intervals]

        fig = go.Figure()

        # Add bars with error bars
        fig.add_trace(
            go.Bar(
                x=[f"Δ={i}" for i in intervals],
                y=mean_corrs,
                error_y=dict(type="data", array=std_corrs),
                marker_color="indianred",
                hovertemplate="Delta: %{x}<br>Mean Correlation: %{y:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Layer Distance (Δ)",
            yaxis_title="Mean Correlation",
            width=800,
            height=500,
            showlegend=False,
            font=dict(size=12),
        )

        # Save to file
        fig.write_html(save_path)
        print(f"Saved periodic pattern analysis to {save_path}")

        # Also create a detailed plot showing all correlations
        if any("all_correlations" in periodic_data[i] for i in intervals):
            self._plot_periodic_pattern_detailed(
                periodic_data, save_path.replace(".html", "_detailed.html")
            )

    def _plot_periodic_pattern_detailed(
        self, periodic_data: Dict[int, Dict[str, Any]], save_path: str
    ):
        """Create a detailed view of all correlations at each interval."""
        intervals = sorted(periodic_data.keys())

        fig = make_subplots(
            rows=1,
            cols=len(intervals),
            subplot_titles=[f"Δ={i}" for i in intervals],
            shared_yaxes=True,
        )

        for idx, interval in enumerate(intervals, 1):
            if "all_correlations" in periodic_data[interval]:
                corrs = periodic_data[interval]["all_correlations"]

                fig.add_trace(
                    go.Box(
                        y=corrs,
                        name=f"Δ={interval}",
                        marker_color="lightblue",
                        showlegend=False,
                    ),
                    row=1,
                    col=idx,
                )

        fig.update_layout(
            title="Detailed Correlation Distribution by Delta",
            height=500,
            width=max(800, len(intervals) * 200),
            font=dict(size=12),
        )

        fig.update_yaxes(title_text="Correlation", row=1, col=1)

        fig.write_html(save_path)
        print(f"Saved detailed periodic pattern analysis to {save_path}")

    def plot_router_similarity_matrix(
        self,
        similarity_data: Dict[str, np.ndarray],
        save_path: str = "router_similarity_matrix.html",
        title: str = "Router Weight Cosine Similarity",
    ):
        """
        Create a heatmap of router weight similarities.

        Args:
            similarity_data: Dictionary from compute_router_weight_similarity()
            save_path: Path to save the HTML file
            title: Plot title
        """
        if "error" in similarity_data:
            print(f"Cannot plot router similarity: {similarity_data['error']}")
            return

        cosine_sim_matrix = similarity_data["cosine_similarity_matrix"]
        router_layers = similarity_data.get(
            "router_layers", list(range(len(cosine_sim_matrix)))
        )

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=cosine_sim_matrix,
                x=[f"Layer {idx}" for idx in router_layers],
                y=[f"Layer {idx}" for idx in router_layers],
                colorscale="Viridis",
                colorbar=dict(title="Cosine<br>Similarity"),
                hovertemplate="Layer %{y} vs Layer %{x}<br>Similarity: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Layer Index",
            yaxis_title="Layer Index",
            width=800,
            height=800,
            font=dict(size=12),
        )

        # Save to file
        fig.write_html(save_path)
        print(f"Saved router similarity matrix to {save_path}")

        # Also plot column norm correlation
        if "column_norm_correlation" in similarity_data:
            self._plot_router_column_norm_correlation(
                similarity_data, save_path.replace(".html", "_column_norms.html")
            )

    def _plot_router_column_norm_correlation(
        self, similarity_data: Dict[str, np.ndarray], save_path: str
    ):
        """Plot correlation of router column norms across deltas."""
        norm_corr = similarity_data["column_norm_correlation"]
        router_layers = similarity_data.get(
            "router_layers", list(range(len(norm_corr)))
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=norm_corr,
                x=[f"Δ={d}" for d in range(1, norm_corr.shape[1] + 1)],
                y=[f"Layer {idx}" for idx in router_layers],
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Correlation"),
                hovertemplate="Layer: %{y}<br>Delta: %{x}<br>Correlation: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Router Column Norm Correlation",
            xaxis_title="Layer Distance (Δ)",
            yaxis_title="Starting Layer",
            width=1000,
            height=600,
            font=dict(size=12),
        )

        fig.write_html(save_path)
        print(f"Saved router column norm correlation to {save_path}")

    def plot_expert_weight_similarity(
        self,
        similarity_data: Dict[str, Any],
        save_path: str = "expert_weight_similarity.html",
        title: Optional[str] = None,
    ):
        """
        Create visualizations for expert weight similarity analysis.

        Args:
            similarity_data: Dictionary from compute_expert_weight_similarity()
            save_path: Path to save the HTML file
            title: Plot title
        """
        if "error" in similarity_data:
            print(f"Cannot plot expert weight similarity: {similarity_data['error']}")
            return

        delta = similarity_data["delta"]
        if title is None:
            title = f"Expert Weight Similarity (Δ={delta})"

        similarities = similarity_data["similarities"]
        similarity_values = [s["similarity"] for s in similarities]

        # Create histogram
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=similarity_values,
                nbinsx=50,
                marker_color="steelblue",
                hovertemplate="Similarity: %{x:.4f}<br>Count: %{y}<extra></extra>",
            )
        )

        # Add vertical lines for mean and median
        mean_sim = similarity_data["mean_similarity"]
        median_sim = similarity_data["median_similarity"]

        fig.add_vline(
            x=mean_sim,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_sim:.4f}",
        )
        fig.add_vline(
            x=median_sim,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_sim:.4f}",
        )

        fig.update_layout(
            title=title,
            xaxis_title="Cosine Similarity",
            yaxis_title="Frequency",
            width=900,
            height=500,
            showlegend=False,
            font=dict(size=12),
        )

        # Save to file
        fig.write_html(save_path)
        print(f"Saved expert weight similarity to {save_path}")

        # Create a detailed scatter plot
        self._plot_expert_similarity_scatter(
            similarities, save_path.replace(".html", "_scatter.html"), delta
        )

    def _plot_expert_similarity_scatter(
        self, similarities: List[Dict], save_path: str, delta: int
    ):
        """Create a scatter plot showing similarity per expert index."""
        expert_indices = [s["expert_idx"] for s in similarities]
        similarity_values = [s["similarity"] for s in similarities]
        layer_pairs = [f"L{s['layer_i']}→L{s['layer_j']}" for s in similarities]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=expert_indices,
                y=similarity_values,
                mode="markers",
                marker=dict(
                    size=8,
                    color=similarity_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Similarity"),
                ),
                text=layer_pairs,
                hovertemplate="Expert: %{x}<br>Similarity: %{y:.4f}<br>%{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Expert Weight Similarity by Expert Index (Δ={delta})",
            xaxis_title="Expert Index",
            yaxis_title="Cosine Similarity",
            width=1000,
            height=500,
            font=dict(size=12),
        )

        fig.write_html(save_path)
        print(f"Saved expert similarity scatter plot to {save_path}")

    def create_comprehensive_report(
        self,
        analyzer,
        output_dir: str = "./moe_analysis_results",
        periodic_intervals: List[int] = [12, 24],
    ):
        """
        Generate a comprehensive analysis report with all visualizations.

        Args:
            analyzer: MoEAnalyzer instance with collected data
            output_dir: Directory to save all outputs
            periodic_intervals: List of deltas to analyze for periodic patterns
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("Generating Comprehensive MoE Analysis Report")
        print("=" * 60)

        # 1. Expert Activation Heatmap
        print("\n[1/7] Generating expert activation heatmap...")
        activation_matrix = analyzer.get_expert_activation_matrix()
        self.plot_expert_activation_heatmap(
            activation_matrix,
            layer_indices=analyzer.layer_indices,
            save_path=os.path.join(output_dir, "expert_activation_heatmap.html"),
        )

        # 1.5. Expert Activation 3D Visualization
        print("\n[2/7] Generating 3D expert activation visualization...")
        try:
            self.plot_expert_activation_3d(
                analyzer,
                save_path=os.path.join(output_dir, "expert_activation_3d.html"),
                max_tokens=50,  # Limit for performance
                max_layers=10,  # Limit for clarity
                max_experts=64,  # Limit for performance
            )
        except Exception as e:
            print(f"⚠️  Failed to generate 3D visualization: {e}")

        # 2. Layer Correlation Matrix
        print("\n[3/7] Computing and plotting layer correlation matrix...")
        correlation_data = analyzer.compute_layer_correlation_matrix(delta_max=30)
        self.plot_layer_correlation_matrix(
            correlation_data,
            save_path=os.path.join(output_dir, "layer_correlation_matrix.html"),
        )

        # 3. Periodic Patterns
        print(
            f"\n[4/7] Analyzing periodic patterns (intervals: {periodic_intervals})..."
        )
        periodic_data = analyzer.compute_periodic_patterns(intervals=periodic_intervals)
        self.plot_periodic_pattern(
            periodic_data,
            save_path=os.path.join(output_dir, "periodic_pattern_analysis.html"),
        )

        # 4. Router Weight Similarity
        print("\n[5/7] Computing router weight similarity...")
        router_sim_data = analyzer.compute_router_weight_similarity()
        self.plot_router_similarity_matrix(
            router_sim_data,
            save_path=os.path.join(output_dir, "router_similarity_matrix.html"),
        )

        # 5. Expert Weight Similarity (可选，耗时较长)
        print(
            f"\n[6/7] Computing expert weight similarity (delta={periodic_intervals[0]})..."
        )
        print("    Using parallel processing to speed up computation...")
        print(
            "    You can press Ctrl+C to skip this step (other results are already saved)"
        )
        try:
            expert_sim_data = analyzer.compute_expert_weight_similarity(
                delta=periodic_intervals[0],
                use_parallel=True,  # 启用并行
                n_jobs=None,  # 使用所有CPU核心
            )
            self.plot_expert_weight_similarity(
                expert_sim_data,
                save_path=os.path.join(output_dir, "expert_weight_similarity.html"),
            )
        except KeyboardInterrupt:
            print("\n    ⏭️  Skipped expert weight similarity computation")
            expert_sim_data = None
        except Exception as e:
            print(f"\n    ⚠️  Error computing expert weight similarity: {e}")
            expert_sim_data = None

        # 6. Summary statistics
        print("\n[7/7] Generating summary report...")
        summary = analyzer.get_summary_statistics()
        self._save_summary_report(summary, periodic_data, output_dir)

        print("\n" + "=" * 60)
        print(f"Analysis complete! Results saved to: {output_dir}")
        print("=" * 60)

    def _save_summary_report(self, summary: Dict, periodic_data: Dict, output_dir: str):
        """Save a text summary of the analysis."""
        report_path = os.path.join(output_dir, "summary_report.txt")

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("MoE Expert Activation Analysis - Summary Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("Model Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of MoE Layers: {summary['num_moe_layers']}\n")
            f.write(f"Layer Indices: {summary['layer_indices']}\n")
            f.write(f"Total Tokens Analyzed: {summary['total_tokens_analyzed']}\n")
            f.write(f"Experts per Layer: {summary['num_experts_per_layer']}\n")
            f.write(f"Top-K per Layer: {summary['top_k_per_layer']}\n")

            f.write("\n\nPeriodic Pattern Analysis:\n")
            f.write("-" * 40 + "\n")
            for interval, data in sorted(periodic_data.items()):
                f.write(f"\nDelta = {interval}:\n")
                f.write(f"  Mean Correlation: {data['mean_correlation']:.4f}\n")
                f.write(f"  Std Correlation: {data['std_correlation']:.4f}\n")
                f.write(f"  Number of Pairs: {data['num_pairs']}\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"Saved summary report to {report_path}")

    def export_structured_data(
        self,
        analyzer,
        output_dir: str,
        format: str = "json",
        include_raw_data: bool = False,
    ):
        """
        Export analysis results as structured data for programmatic access.

        Args:
            analyzer: MoEAnalyzer instance with collected data
            output_dir: Output directory
            format: Output format - "json", "jsonl", or "pickle"
            include_raw_data: Whether to include raw activation matrices (can be large)
        """
        print(f"\nExporting structured data (format: {format})...")

        # Collect all analysis results
        structured_data = {
            "metadata": {
                "model_info": analyzer.get_summary_statistics(),
                "export_format": format,
                "include_raw_data": include_raw_data,
            },
            "expert_activation": {},
            "layer_correlation": {},
            "periodic_patterns": {},
            "router_similarity": {},
        }

        # 1. Expert Activation Statistics
        if hasattr(analyzer, "activation_stats") and analyzer.activation_stats:
            structured_data["expert_activation"] = {
                "matrix": analyzer.get_expert_activation_matrix().tolist(),
                "layer_indices": analyzer.layer_indices,
                "per_layer_stats": {},
            }

            for layer_idx, stats in analyzer.activation_stats.items():
                layer_data = {
                    "avg_activation": stats["avg_activation"].tolist(),
                    "var_activation": stats["var_activation"].tolist(),
                    "num_tokens": int(stats["num_tokens"]),
                    "num_experts": int(stats["num_experts"]),
                }

                if include_raw_data:
                    layer_data["routing_probs"] = (
                        stats["routing_probs"].numpy().tolist()
                    )
                    layer_data["selected_experts"] = (
                        stats["selected_experts"].numpy().tolist()
                    )

                structured_data["expert_activation"]["per_layer_stats"][
                    str(layer_idx)
                ] = layer_data

        # 2. Layer Correlation Analysis
        print("    Computing layer correlations...")
        try:
            corr_data = analyzer.compute_layer_correlation_matrix(delta_max=30)
            structured_data["layer_correlation"] = {
                "correlation_matrix": self._convert_nan_to_none(
                    corr_data["correlation_matrix"].tolist()
                ),
                "deltas": corr_data["deltas"],
                "layer_indices": corr_data["layer_indices"],
                "layer_pairs": [
                    {
                        "layer_i": int(p[0]),
                        "layer_j": int(p[1]),
                        "delta": int(p[2]),
                        "correlation": float(p[3]) if not np.isnan(p[3]) else None,
                    }
                    for p in corr_data["layer_pairs"]
                ],
            }
        except Exception as e:
            structured_data["layer_correlation"]["error"] = str(e)

        # 3. Periodic Pattern Analysis
        print("    Analyzing periodic patterns...")
        try:
            periodic_data = analyzer.compute_periodic_patterns(intervals=[12, 24, 36])
            structured_data["periodic_patterns"] = {
                str(interval): {
                    "mean_correlation": float(data["mean_correlation"]),
                    "std_correlation": float(data["std_correlation"]),
                    "num_pairs": int(data["num_pairs"]),
                    "all_correlations": [
                        float(c) for c in data.get("all_correlations", [])
                    ],
                }
                for interval, data in periodic_data.items()
            }
        except Exception as e:
            structured_data["periodic_patterns"]["error"] = str(e)

        # 4. Router Similarity
        print("    Computing router similarities...")
        try:
            router_sim = analyzer.compute_router_weight_similarity()
            if "error" not in router_sim:
                structured_data["router_similarity"] = {
                    "cosine_similarity_matrix": router_sim[
                        "cosine_similarity_matrix"
                    ].tolist(),
                    "column_norm_correlation": self._convert_nan_to_none(
                        router_sim["column_norm_correlation"].tolist()
                    ),
                    "router_layers": router_sim["router_layers"],
                }
            else:
                structured_data["router_similarity"]["error"] = router_sim["error"]
        except Exception as e:
            structured_data["router_similarity"]["error"] = str(e)

        # Export based on format
        if format == "json":
            output_path = os.path.join(output_dir, "analysis_data.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved structured data to {output_path}")

        elif format == "jsonl":
            output_path = os.path.join(output_dir, "analysis_data.jsonl")
            with open(output_path, "w", encoding="utf-8") as f:
                for key, value in structured_data.items():
                    json.dump({key: value}, f, ensure_ascii=False)
                    f.write("\n")
            print(f"✓ Saved structured data to {output_path}")

        elif format == "pickle":
            output_path = os.path.join(output_dir, "analysis_data.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(structured_data, f)
            print(f"✓ Saved structured data to {output_path}")

        # Also save a compact summary JSON
        summary_path = os.path.join(output_dir, "analysis_summary.json")
        summary = {
            "model": structured_data["metadata"]["model_info"],
            "key_findings": {
                "num_moe_layers": (
                    len(analyzer.layer_indices)
                    if hasattr(analyzer, "layer_indices")
                    else 0
                ),
                "periodic_patterns": {
                    k: {
                        "mean_correlation": v["mean_correlation"],
                        "strength": (
                            "strong"
                            if v["mean_correlation"] > 0.7
                            else "moderate" if v["mean_correlation"] > 0.4 else "weak"
                        ),
                    }
                    for k, v in structured_data.get("periodic_patterns", {}).items()
                    if isinstance(v, dict) and "mean_correlation" in v
                },
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved analysis summary to {summary_path}")

        return structured_data

    def _convert_nan_to_none(self, data):
        """Convert NaN values to None for JSON serialization."""
        if isinstance(data, list):
            return [self._convert_nan_to_none(item) for item in data]
        elif isinstance(data, float) and np.isnan(data):
            return None
        else:
            return data
