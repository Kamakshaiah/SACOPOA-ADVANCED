"""
ML Visualization utilities for SACOPOA ML Module
===============================================

Visualization tools for machine learning results,
performance analytics, and educational insights.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MLVisualizer:
    """
    Visualization utilities for SACOPOA machine learning results.
    
    Capabilities:
    - Performance prediction visualizations
    - Risk assessment charts
    - Clustering analysis plots
    - Intervention recommendation displays
    - Anomaly detection visualizations
    """
    
    def __init__(self, style: str = 'educational'):
        """
        Initialize the ML visualizer.
        
        Args:
            style: Visualization style ('educational', 'professional', 'minimal')
        """
        self.style = style
        self.color_palette = self._get_color_palette()
        self.fig_size = (12, 8) if style == 'professional' else (10, 6)
        
    def _get_color_palette(self) -> List[str]:
        """Get color palette based on style."""
        if self.style == 'educational':
            return ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0']
        elif self.style == 'professional':
            return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        else:  # minimal
            return ['#333333', '#666666', '#999999', '#CCCCCC', '#E0E0E0', '#F5F5F5']
    
    def plot_performance_predictions(self, actual_scores: np.ndarray, 
                                   predicted_scores: np.ndarray,
                                   prediction_intervals: Dict = None,
                                   student_ids: List[str] = None) -> go.Figure:
        """
        Plot actual vs predicted performance scores.
        
        Args:
            actual_scores: Actual performance scores
            predicted_scores: Predicted performance scores
            prediction_intervals: Optional prediction intervals
            student_ids: Optional student identifiers
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add prediction intervals if available
        if prediction_intervals:
            fig.add_trace(go.Scatter(
                x=list(range(len(predicted_scores))),
                y=prediction_intervals.get('upper_bound', []),
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.2)',
                name='Prediction Interval',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(predicted_scores))),
                y=prediction_intervals.get('lower_bound', []),
                mode='lines',
                fill='tonexty',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.2)',
                name='95% Confidence Interval',
                showlegend=True
            ))
        
        # Add actual scores
        fig.add_trace(go.Scatter(
            x=list(range(len(actual_scores))),
            y=actual_scores,
            mode='markers',
            marker=dict(
                color=self.color_palette[0],
                size=8,
                symbol='circle'
            ),
            name='Actual Scores',
            text=student_ids if student_ids else [f'Student {i+1}' for i in range(len(actual_scores))],
            hovertemplate='<b>%{text}</b><br>Actual: %{y:.1f}<extra></extra>'
        ))
        
        # Add predicted scores
        fig.add_trace(go.Scatter(
            x=list(range(len(predicted_scores))),
            y=predicted_scores,
            mode='markers',
            marker=dict(
                color=self.color_palette[1],
                size=8,
                symbol='diamond'
            ),
            name='Predicted Scores',
            text=student_ids if student_ids else [f'Student {i+1}' for i in range(len(predicted_scores))],
            hovertemplate='<b>%{text}</b><br>Predicted: %{y:.1f}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_score = min(np.min(actual_scores), np.min(predicted_scores))
        max_score = max(np.max(actual_scores), np.max(predicted_scores))
        
        fig.add_trace(go.Scatter(
            x=[min_score, max_score],
            y=[min_score, max_score],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Perfect Prediction',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Performance Prediction Results',
            xaxis_title='Student Index' if not student_ids else 'Students',
            yaxis_title='Performance Score (%)',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_risk_assessment(self, risk_probabilities: np.ndarray,
                           risk_categories: List[str],
                           student_ids: List[str] = None) -> go.Figure:
        """
        Plot risk assessment results.
        
        Args:
            risk_probabilities: Risk probabilities for each student
            risk_categories: Risk category labels
            student_ids: Optional student identifiers
            
        Returns:
            Plotly figure object
        """
        # Create color mapping for risk levels
        risk_colors = {
            'Low Risk': '#4CAF50',
            'Medium Risk': '#FF9800', 
            'High Risk': '#FF5722',
            'Critical Risk': '#F44336'
        }
        
        colors = [risk_colors.get(cat, '#666666') for cat in risk_categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=student_ids if student_ids else [f'Student {i+1}' for i in range(len(risk_probabilities))],
            y=risk_probabilities,
            marker_color=colors,
            text=[f'{cat}<br>{prob:.2f}' for cat, prob in zip(risk_categories, risk_probabilities)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Risk Level: %{text}<br>Probability: %{y:.3f}<extra></extra>'
        ))
        
        # Add risk threshold lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Risk Threshold")
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        
        fig.update_layout(
            title='Student Risk Assessment',
            xaxis_title='Students',
            yaxis_title='Risk Probability',
            template='plotly_white',
            height=600,
            xaxis_tickangle=-45 if student_ids else 0
        )
        
        return fig
    
    def plot_clustering_results(self, features: np.ndarray, 
                              cluster_labels: np.ndarray,
                              cluster_centers: np.ndarray = None,
                              feature_names: List[str] = None) -> go.Figure:
        """
        Plot clustering results using PCA for visualization.
        
        Args:
            features: Feature matrix
            cluster_labels: Cluster assignment for each point
            cluster_centers: Optional cluster centers
            feature_names: Optional feature names
            
        Returns:
            Plotly figure object
        """
        from sklearn.decomposition import PCA
        
        # Reduce to 2D using PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        unique_clusters = np.unique(cluster_labels)
        
        fig = go.Figure()
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = features_2d[cluster_mask]
            
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                marker=dict(
                    color=self.color_palette[i % len(self.color_palette)],
                    size=8,
                    opacity=0.7
                ),
                name=f'Cluster {cluster_id}',
                hovertemplate=f'<b>Cluster {cluster_id}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
            ))
        
        # Plot cluster centers if available
        if cluster_centers is not None:
            centers_2d = pca.transform(cluster_centers)
            fig.add_trace(go.Scatter(
                x=centers_2d[:, 0],
                y=centers_2d[:, 1],
                mode='markers',
                marker=dict(
                    color='black',
                    size=15,
                    symbol='x',
                    line=dict(width=2, color='white')
                ),
                name='Cluster Centers',
                hovertemplate='<b>Cluster Center</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
        
        # Add explained variance information
        explained_var = pca.explained_variance_ratio_
        
        fig.update_layout(
            title=f'Student Performance Clustering (PCA Projection)<br><sub>PC1: {explained_var[0]:.1%} variance, PC2: {explained_var[1]:.1%} variance</sub>',
            xaxis_title=f'Principal Component 1 ({explained_var[0]:.1%} variance)',
            yaxis_title=f'Principal Component 2 ({explained_var[1]:.1%} variance)',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def plot_intervention_recommendations(self, recommendations: List[Dict],
                                        top_n: int = 10) -> go.Figure:
        """
        Plot intervention recommendation analysis.
        
        Args:
            recommendations: List of intervention recommendations
            top_n: Number of top interventions to display
            
        Returns:
            Plotly figure object
        """
        # Count intervention frequencies
        intervention_counts = {}
        intervention_urgency = {}
        
        for rec in recommendations:
            for intervention in rec['recommendations']:
                int_type = intervention['type']
                intervention_counts[int_type] = intervention_counts.get(int_type, 0) + 1
                
                # Track urgency levels
                urgency = intervention.get('urgency', 'medium')
                if int_type not in intervention_urgency:
                    intervention_urgency[int_type] = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                intervention_urgency[int_type][urgency] += 1
        
        # Get top interventions
        top_interventions = sorted(intervention_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create stacked bar chart
        interventions = [item[0] for item in top_interventions]
        urgency_levels = ['critical', 'high', 'medium', 'low']
        urgency_colors = {'critical': '#F44336', 'high': '#FF9800', 'medium': '#FFC107', 'low': '#4CAF50'}
        
        fig = go.Figure()
        
        for urgency in urgency_levels:
            counts = [intervention_urgency.get(intervention, {}).get(urgency, 0) for intervention in interventions]
            
            fig.add_trace(go.Bar(
                name=urgency.capitalize(),
                x=interventions,
                y=counts,
                marker_color=urgency_colors[urgency],
                hovertemplate=f'<b>%{{x}}</b><br>{urgency.capitalize()} urgency: %{{y}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Intervention Recommendations by Type and Urgency',
            xaxis_title='Intervention Type',
            yaxis_title='Number of Recommendations',
            barmode='stack',
            template='plotly_white',
            height=600,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_anomaly_detection(self, features: np.ndarray,
                             anomaly_mask: np.ndarray,
                             anomaly_scores: np.ndarray = None,
                             student_ids: List[str] = None) -> go.Figure:
        """
        Plot anomaly detection results.
        
        Args:
            features: Feature matrix
            anomaly_mask: Boolean mask indicating anomalies
            anomaly_scores: Optional anomaly scores
            student_ids: Optional student identifiers
            
        Returns:
            Plotly figure object
        """
        from sklearn.decomposition import PCA
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        normal_points = features_2d[~anomaly_mask]
        anomaly_points = features_2d[anomaly_mask]
        
        fig = go.Figure()
        
        # Plot normal points
        fig.add_trace(go.Scatter(
            x=normal_points[:, 0],
            y=normal_points[:, 1],
            mode='markers',
            marker=dict(
                color=self.color_palette[0],
                size=6,
                opacity=0.6
            ),
            name='Normal Students',
            hovertemplate='<b>Normal Student</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ))
        
        # Plot anomalous points
        anomaly_colors = anomaly_scores[anomaly_mask] if anomaly_scores is not None else 'red'
        
        fig.add_trace(go.Scatter(
            x=anomaly_points[:, 0],
            y=anomaly_points[:, 1],
            mode='markers',
            marker=dict(
                color=anomaly_colors,
                colorscale='Reds',
                size=10,
                opacity=0.8,
                colorbar=dict(title="Anomaly Score") if anomaly_scores is not None else None
            ),
            name='Anomalous Students',
            text=[student_ids[i] for i in np.where(anomaly_mask)[0]] if student_ids else None,
            hovertemplate='<b>Anomalous Student</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ))
        
        explained_var = pca.explained_variance_ratio_
        
        fig.update_layout(
            title=f'Anomaly Detection Results (PCA Projection)<br><sub>PC1: {explained_var[0]:.1%} variance, PC2: {explained_var[1]:.1%} variance</sub>',
            xaxis_title=f'Principal Component 1 ({explained_var[0]:.1%} variance)',
            yaxis_title=f'Principal Component 2 ({explained_var[1]:.1%} variance)',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def create_ml_dashboard(self, ml_results: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive ML dashboard with multiple visualizations.
        
        Args:
            ml_results: Dictionary containing all ML analysis results
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Predictions', 'Risk Assessment', 
                          'Clustering Analysis', 'Anomaly Detection'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Performance predictions (if available)
        if 'predictions' in ml_results:
            pred_data = ml_results['predictions']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pred_data['actual']))),
                    y=pred_data['actual'],
                    mode='markers',
                    name='Actual',
                    marker=dict(color=self.color_palette[0])
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pred_data['predicted']))),
                    y=pred_data['predicted'],
                    mode='markers',
                    name='Predicted',
                    marker=dict(color=self.color_palette[1])
                ),
                row=1, col=1
            )
        
        # Risk assessment (if available)
        if 'risk_assessment' in ml_results:
            risk_data = ml_results['risk_assessment']
            risk_categories = risk_data.get('risk_categories', [])
            category_counts = {cat: risk_categories.count(cat) for cat in set(risk_categories)}
            
            fig.add_trace(
                go.Bar(
                    x=list(category_counts.keys()),
                    y=list(category_counts.values()),
                    name='Risk Distribution',
                    marker=dict(color=self.color_palette[2])
                ),
                row=1, col=2
            )
        
        # Clustering (if available)
        if 'clustering' in ml_results:
            cluster_data = ml_results['clustering']
            features_2d = cluster_data.get('features_2d')
            cluster_labels = cluster_data.get('cluster_labels')
            
            if features_2d is not None and cluster_labels is not None:
                unique_clusters = np.unique(cluster_labels)
                for i, cluster_id in enumerate(unique_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_points = features_2d[cluster_mask]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=cluster_points[:, 0],
                            y=cluster_points[:, 1],
                            mode='markers',
                            name=f'Cluster {cluster_id}',
                            marker=dict(color=self.color_palette[i % len(self.color_palette)])
                        ),
                        row=2, col=1
                    )
        
        # Anomaly detection (if available)
        if 'anomalies' in ml_results:
            anomaly_data = ml_results['anomalies']
            features_2d = anomaly_data.get('features_2d')
            anomaly_mask = anomaly_data.get('anomaly_mask')
            
            if features_2d is not None and anomaly_mask is not None:
                normal_points = features_2d[~anomaly_mask]
                anomaly_points = features_2d[anomaly_mask]
                
                fig.add_trace(
                    go.Scatter(
                        x=normal_points[:, 0],
                        y=normal_points[:, 1],
                        mode='markers',
                        name='Normal',
                        marker=dict(color=self.color_palette[0], size=4)
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_points[:, 0],
                        y=anomaly_points[:, 1],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8)
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="SACOPOA Machine Learning Analysis Dashboard",
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save figure to file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Output format ('html', 'png', 'pdf', 'svg')
        """
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        elif format == 'svg':
            fig.write_image(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def create_static_plots(self, ml_results: Dict[str, Any], save_dir: str = './'):
        """
        Create and save static matplotlib plots for reports.
        
        Args:
            ml_results: ML analysis results
            save_dir: Directory to save plots
        """
        # Set up matplotlib style
        plt.rcParams['figure.figsize'] = self.fig_size
        plt.rcParams['font.size'] = 10
        
        # Performance prediction plot
        if 'predictions' in ml_results:
            self._create_static_prediction_plot(ml_results['predictions'], save_dir)
        
        # Risk assessment plot  
        if 'risk_assessment' in ml_results:
            self._create_static_risk_plot(ml_results['risk_assessment'], save_dir)
        
        # Clustering plot
        if 'clustering' in ml_results:
            self._create_static_clustering_plot(ml_results['clustering'], save_dir)
    
    def _create_static_prediction_plot(self, pred_data: Dict, save_dir: str):
        """Create static prediction plot."""
        plt.figure(figsize=self.fig_size)
        
        actual = pred_data['actual']
        predicted = pred_data['predicted']
        
        plt.scatter(actual, predicted, alpha=0.6, color=self.color_palette[0])
        
        # Perfect prediction line
        min_val = min(np.min(actual), np.min(predicted))
        max_val = max(np.max(actual), np.max(predicted))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('Actual Performance (%)')
        plt.ylabel('Predicted Performance (%)')
        plt.title('Performance Prediction Results')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_static_risk_plot(self, risk_data: Dict, save_dir: str):
        """Create static risk assessment plot."""
        plt.figure(figsize=self.fig_size)
        
        risk_categories = risk_data.get('risk_categories', [])
        category_counts = {cat: risk_categories.count(cat) for cat in set(risk_categories)}
        
        colors = ['#4CAF50', '#FF9800', '#FF5722', '#F44336'][:len(category_counts)]
        
        plt.pie(category_counts.values(), labels=category_counts.keys(), 
                colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Risk Assessment Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/risk_assessment.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_static_clustering_plot(self, cluster_data: Dict, save_dir: str):
        """Create static clustering plot."""
        features_2d = cluster_data.get('features_2d')
        cluster_labels = cluster_data.get('cluster_labels')
        
        if features_2d is None or cluster_labels is None:
            return
        
        plt.figure(figsize=self.fig_size)
        
        unique_clusters = np.unique(cluster_labels)
        colors = self.color_palette[:len(unique_clusters)]
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = features_2d[cluster_mask]
            
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i], label=f'Cluster {cluster_id}', alpha=0.7)
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Student Performance Clustering')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()