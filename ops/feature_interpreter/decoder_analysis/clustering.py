"""
Feature Clustering Module.

This module provides functionality for clustering features based on 
their decoder weights and similarities.
"""

import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

# Configure logging
logger = logging.getLogger(__name__)

def cluster_features(
    feature_data: Dict,
    method: str = "kmeans",
    n_clusters: int = 5,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Cluster features into model-specific vs. shared features groups.
    
    Args:
        feature_data: Dictionary with feature data (from extract_feature_decoder_norms)
        method: Clustering method ('kmeans', 'hierarchical', or 'dbscan')
        n_clusters: Number of clusters for KMeans or hierarchical clustering
        output_dir: Optional directory to save visualizations
        
    Returns:
        Dictionary with clustering results
    """
    logger.info(f"Clustering features using {method} method")
    
    if "feature_norms" not in feature_data or "feature_decoders" not in feature_data:
        raise ValueError("Feature data must contain 'feature_norms' and 'feature_decoders'")
    
    # Extract feature IDs and create feature vectors
    feature_ids = list(feature_data["feature_norms"].keys())
    
    # Create feature vectors with [base_norm, target_norm]
    feature_vectors = np.array([
        [
            feature_data["feature_norms"][f_id]["base_norm"],
            feature_data["feature_norms"][f_id]["target_norm"]
        ]
        for f_id in feature_ids
    ])
    
    # Apply log transformation to handle large value differences
    feature_vectors = np.log1p(feature_vectors)
    
    # Initialize clustering results
    clusters = {}
    
    if method == "kmeans":
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_vectors)
        
        # Store cluster centers
        clusters["centers"] = kmeans.cluster_centers_.tolist()
        
    elif method == "hierarchical":
        # Apply hierarchical clustering
        distances = pdist(feature_vectors)
        linkage_matrix = linkage(distances, method='ward')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        # No centers in hierarchical clustering
        clusters["centers"] = None
        
    elif method == "dbscan":
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(feature_vectors)
        
        # No centers in DBSCAN
        clusters["centers"] = None
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    # Group features by cluster
    clusters["feature_clusters"] = {}
    for i, cluster_id in enumerate(cluster_labels):
        cluster_id_str = str(cluster_id)
        if cluster_id_str not in clusters["feature_clusters"]:
            clusters["feature_clusters"][cluster_id_str] = []
        
        feature_id = feature_ids[i]
        clusters["feature_clusters"][cluster_id_str].append({
            "id": feature_id,
            "base_norm": feature_data["feature_norms"][feature_id]["base_norm"],
            "target_norm": feature_data["feature_norms"][feature_id]["target_norm"],
            "norm_ratio": feature_data["feature_norms"][feature_id]["norm_ratio"]
        })
    
    # Create visualizations if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot feature norm scatter with clusters
        plt.figure(figsize=(10, 8))
        
        # Plot points colored by cluster
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            plt.scatter(
                feature_vectors[mask, 0],
                feature_vectors[mask, 1],
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )
        
        # Plot cluster centers for KMeans
        if method == "kmeans":
            centers = np.array(clusters["centers"])
            plt.scatter(
                centers[:, 0],
                centers[:, 1],
                marker='*',
                s=200,
                color='black',
                label='Cluster Centers'
            )
        
        plt.xlabel('Log(Base Norm + 1)')
        plt.ylabel('Log(Target Norm + 1)')
        plt.title('Feature Clustering by Decoder Norms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'feature_clusters_{method}.png', dpi=300)
        plt.close()
        
        # Save clustering results
        with open(output_dir / f'feature_clusters_{method}.json', 'w') as f:
            json.dump(clusters, f, indent=2)
        
        logger.info(f"Saved clustering results to {output_dir}")
        
        # If we have more than 3 features, create dimensionality reduction visualization
        if len(feature_ids) > 3:
            try:
                # Extract full decoder weights for PCA/t-SNE visualization
                feature_decoders = []
                for f_id in feature_ids:
                    # Concatenate base and target decoders
                    base_decoder = np.array(feature_data["feature_decoders"][f_id]["base_decoder"])
                    target_decoder = np.array(feature_data["feature_decoders"][f_id]["target_decoder"])
                    feature_decoders.append(np.concatenate([base_decoder, target_decoder]))
                
                feature_decoders = np.array(feature_decoders)
                
                # Apply PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(feature_decoders)
                
                # Plot PCA results
                plt.figure(figsize=(10, 8))
                for cluster_id in np.unique(cluster_labels):
                    mask = cluster_labels == cluster_id
                    plt.scatter(
                        pca_result[mask, 0],
                        pca_result[mask, 1],
                        label=f'Cluster {cluster_id}',
                        alpha=0.7
                    )
                
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.title('PCA of Feature Decoder Weights (Colored by Clusters)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'feature_clusters_pca.png', dpi=300)
                plt.close()
                
                # Apply t-SNE (if we have more than a handful of features)
                if len(feature_ids) > 10:
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(feature_decoders)
                    
                    # Plot t-SNE results
                    plt.figure(figsize=(10, 8))
                    for cluster_id in np.unique(cluster_labels):
                        mask = cluster_labels == cluster_id
                        plt.scatter(
                            tsne_result[mask, 0],
                            tsne_result[mask, 1],
                            label=f'Cluster {cluster_id}',
                            alpha=0.7
                        )
                    
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.title('t-SNE of Feature Decoder Weights (Colored by Clusters)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(output_dir / 'feature_clusters_tsne.png', dpi=300)
                    plt.close()
                
                logger.info("Created dimensionality reduction visualizations for feature clusters")
            except Exception as e:
                logger.error(f"Failed to create dimensionality reduction visualizations: {e}")
    
    return clusters 