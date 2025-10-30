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


class MoEVisualizer:
    """
    Creates interactive visualizations for MoE analysis results.
    
    All plots are generated as HTML files that can be opened in a browser.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def plot_expert_activation_heatmap(
        self,
        activation_matrix: np.ndarray,
        layer_indices: Optional[List[int]] = None,
        save_path: str = "expert_activation_heatmap.html",
        title: str = "Expert Activation Probability Heatmap"
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
        fig = go.Figure(data=go.Heatmap(
            z=activation_matrix,
            x=[f"Expert {i}" for i in range(num_experts)],
            y=[f"Layer {idx}" for idx in layer_indices],
            colorscale='Viridis',
            colorbar=dict(title="Avg Activation<br>Probability"),
            hovertemplate='Layer: %{y}<br>Expert: %{x}<br>Activation: %{z:.4f}<extra></extra>'
        ))
        
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
        title: str = "Layer-to-Layer Correlation Matrix"
    ):
        """
        Create a heatmap showing correlation between layers at different deltas.
        
        Args:
            correlation_data: Dictionary from compute_layer_correlation_matrix()
            save_path: Path to save the HTML file
            title: Plot title
        """
        correlation_matrix = correlation_data['correlation_matrix']
        layer_indices = correlation_data['layer_indices']
        deltas = correlation_data['deltas']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=[f"Δ={d}" for d in deltas],
            y=[f"Layer {idx}" for idx in layer_indices],
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Pearson<br>Correlation"),
            hovertemplate='Start Layer: %{y}<br>Delta: %{x}<br>Correlation: %{z:.4f}<extra></extra>'
        ))
        
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
        title: str = "Periodic Pattern Analysis"
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
        mean_corrs = [periodic_data[i]['mean_correlation'] for i in intervals]
        std_corrs = [periodic_data[i]['std_correlation'] for i in intervals]
        
        fig = go.Figure()
        
        # Add bars with error bars
        fig.add_trace(go.Bar(
            x=[f"Δ={i}" for i in intervals],
            y=mean_corrs,
            error_y=dict(type='data', array=std_corrs),
            marker_color='indianred',
            hovertemplate='Delta: %{x}<br>Mean Correlation: %{y:.4f}<extra></extra>'
        ))
        
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
        if any('all_correlations' in periodic_data[i] for i in intervals):
            self._plot_periodic_pattern_detailed(periodic_data, save_path.replace('.html', '_detailed.html'))
    
    def _plot_periodic_pattern_detailed(
        self,
        periodic_data: Dict[int, Dict[str, Any]],
        save_path: str
    ):
        """Create a detailed view of all correlations at each interval."""
        intervals = sorted(periodic_data.keys())
        
        fig = make_subplots(
            rows=1, cols=len(intervals),
            subplot_titles=[f"Δ={i}" for i in intervals],
            shared_yaxes=True
        )
        
        for idx, interval in enumerate(intervals, 1):
            if 'all_correlations' in periodic_data[interval]:
                corrs = periodic_data[interval]['all_correlations']
                
                fig.add_trace(
                    go.Box(
                        y=corrs,
                        name=f"Δ={interval}",
                        marker_color='lightblue',
                        showlegend=False
                    ),
                    row=1, col=idx
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
        title: str = "Router Weight Cosine Similarity"
    ):
        """
        Create a heatmap of router weight similarities.
        
        Args:
            similarity_data: Dictionary from compute_router_weight_similarity()
            save_path: Path to save the HTML file
            title: Plot title
        """
        if 'error' in similarity_data:
            print(f"Cannot plot router similarity: {similarity_data['error']}")
            return
        
        cosine_sim_matrix = similarity_data['cosine_similarity_matrix']
        router_layers = similarity_data.get('router_layers', list(range(len(cosine_sim_matrix))))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cosine_sim_matrix,
            x=[f"Layer {idx}" for idx in router_layers],
            y=[f"Layer {idx}" for idx in router_layers],
            colorscale='Viridis',
            colorbar=dict(title="Cosine<br>Similarity"),
            hovertemplate='Layer %{y} vs Layer %{x}<br>Similarity: %{z:.4f}<extra></extra>'
        ))
        
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
        if 'column_norm_correlation' in similarity_data:
            self._plot_router_column_norm_correlation(
                similarity_data,
                save_path.replace('.html', '_column_norms.html')
            )
    
    def _plot_router_column_norm_correlation(
        self,
        similarity_data: Dict[str, np.ndarray],
        save_path: str
    ):
        """Plot correlation of router column norms across deltas."""
        norm_corr = similarity_data['column_norm_correlation']
        router_layers = similarity_data.get('router_layers', list(range(len(norm_corr))))
        
        fig = go.Figure(data=go.Heatmap(
            z=norm_corr,
            x=[f"Δ={d}" for d in range(1, norm_corr.shape[1] + 1)],
            y=[f"Layer {idx}" for idx in router_layers],
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation"),
            hovertemplate='Layer: %{y}<br>Delta: %{x}<br>Correlation: %{z:.4f}<extra></extra>'
        ))
        
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
        title: Optional[str] = None
    ):
        """
        Create visualizations for expert weight similarity analysis.
        
        Args:
            similarity_data: Dictionary from compute_expert_weight_similarity()
            save_path: Path to save the HTML file
            title: Plot title
        """
        if 'error' in similarity_data:
            print(f"Cannot plot expert weight similarity: {similarity_data['error']}")
            return
        
        delta = similarity_data['delta']
        if title is None:
            title = f"Expert Weight Similarity (Δ={delta})"
        
        similarities = similarity_data['similarities']
        similarity_values = [s['similarity'] for s in similarities]
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=similarity_values,
            nbinsx=50,
            marker_color='steelblue',
            hovertemplate='Similarity: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add vertical lines for mean and median
        mean_sim = similarity_data['mean_similarity']
        median_sim = similarity_data['median_similarity']
        
        fig.add_vline(x=mean_sim, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_sim:.4f}")
        fig.add_vline(x=median_sim, line_dash="dot", line_color="green",
                     annotation_text=f"Median: {median_sim:.4f}")
        
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
        self._plot_expert_similarity_scatter(similarities, save_path.replace('.html', '_scatter.html'), delta)
    
    def _plot_expert_similarity_scatter(
        self,
        similarities: List[Dict],
        save_path: str,
        delta: int
    ):
        """Create a scatter plot showing similarity per expert index."""
        expert_indices = [s['expert_idx'] for s in similarities]
        similarity_values = [s['similarity'] for s in similarities]
        layer_pairs = [f"L{s['layer_i']}→L{s['layer_j']}" for s in similarities]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=expert_indices,
            y=similarity_values,
            mode='markers',
            marker=dict(
                size=8,
                color=similarity_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Similarity")
            ),
            text=layer_pairs,
            hovertemplate='Expert: %{x}<br>Similarity: %{y:.4f}<br>%{text}<extra></extra>'
        ))
        
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
        periodic_intervals: List[int] = [12, 24]
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
        
        print("\n" + "="*60)
        print("Generating Comprehensive MoE Analysis Report")
        print("="*60)
        
        # 1. Expert Activation Heatmap
        print("\n[1/6] Generating expert activation heatmap...")
        activation_matrix = analyzer.get_expert_activation_matrix()
        self.plot_expert_activation_heatmap(
            activation_matrix,
            layer_indices=analyzer.layer_indices,
            save_path=os.path.join(output_dir, "expert_activation_heatmap.html")
        )
        
        # 2. Layer Correlation Matrix
        print("\n[2/6] Computing and plotting layer correlation matrix...")
        correlation_data = analyzer.compute_layer_correlation_matrix(delta_max=30)
        self.plot_layer_correlation_matrix(
            correlation_data,
            save_path=os.path.join(output_dir, "layer_correlation_matrix.html")
        )
        
        # 3. Periodic Patterns
        print(f"\n[3/6] Analyzing periodic patterns (intervals: {periodic_intervals})...")
        periodic_data = analyzer.compute_periodic_patterns(intervals=periodic_intervals)
        self.plot_periodic_pattern(
            periodic_data,
            save_path=os.path.join(output_dir, "periodic_pattern_analysis.html")
        )
        
        # 4. Router Weight Similarity
        print("\n[4/6] Computing router weight similarity...")
        router_sim_data = analyzer.compute_router_weight_similarity()
        self.plot_router_similarity_matrix(
            router_sim_data,
            save_path=os.path.join(output_dir, "router_similarity_matrix.html")
        )
        
        # 5. Expert Weight Similarity
        print(f"\n[5/6] Computing expert weight similarity (delta={periodic_intervals[0]})...")
        expert_sim_data = analyzer.compute_expert_weight_similarity(delta=periodic_intervals[0])
        self.plot_expert_weight_similarity(
            expert_sim_data,
            save_path=os.path.join(output_dir, "expert_weight_similarity.html")
        )
        
        # 6. Summary statistics
        print("\n[6/6] Generating summary report...")
        summary = analyzer.get_summary_statistics()
        self._save_summary_report(summary, periodic_data, output_dir)
        
        print("\n" + "="*60)
        print(f"Analysis complete! Results saved to: {output_dir}")
        print("="*60)
    
    def _save_summary_report(
        self,
        summary: Dict,
        periodic_data: Dict,
        output_dir: str
    ):
        """Save a text summary of the analysis."""
        report_path = os.path.join(output_dir, "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MoE Expert Activation Analysis - Summary Report\n")
            f.write("="*60 + "\n\n")
            
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
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"Saved summary report to {report_path}")

