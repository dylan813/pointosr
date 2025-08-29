import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)

class GeometryFeatureExtractor:
    
    def __init__(self, config):
        self.normal_radius = config.geometry.normal_radius
        self.eigen_radius = config.geometry.eigen_radii[0]
        self.min_neighbors = config.geometry.min_neighbors
        self.density_adaptive = getattr(config.geometry, 'density_adaptive', True)
        self.base_density = getattr(config.geometry, 'base_density', 1000)
        self.min_radius = getattr(config.geometry, 'min_radius', 0.05)
        self.max_radius = getattr(config.geometry, 'max_radius', 0.5)
        self.gravity = np.array([0, 0, -1])
        
        logger.info(f"GeometryFeatureExtractor initialized:")
        logger.info(f"  Normal radius: {self.normal_radius}")
        logger.info(f"  Eigen radius: {self.eigen_radius}")
        logger.info(f"  Min neighbors: {self.min_neighbors}")
        logger.info(f"  Density adaptive: {self.density_adaptive}")
        if self.density_adaptive:
            logger.info(f"  Base density: {self.base_density} points/m³")
            logger.info(f"  Radius range: [{self.min_radius}, {self.max_radius}]")
    
    def estimate_point_density(self, points):
        if len(points) < 10:
            return self.base_density
            
        probe_radius = 0.1
        nbrs = NearestNeighbors(radius=probe_radius, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.radius_neighbors(points, radius=probe_radius)
        
        neighbor_counts = [len(idx) for idx in indices]
        avg_neighbors = np.mean(neighbor_counts)
        
        sphere_volume = (4/3) * np.pi * (probe_radius ** 3)
        density = avg_neighbors / sphere_volume
        
        return max(density, 1.0)
    
    def compute_adaptive_radius(self, points, base_radius):
        if not self.density_adaptive:
            return base_radius
            
        density = self.estimate_point_density(points)
        
        density_ratio = self.base_density / density
        adaptive_factor = np.sqrt(density_ratio)
        
        adaptive_radius = base_radius * adaptive_factor
        adaptive_radius = np.clip(adaptive_radius, self.min_radius, self.max_radius)
        
        logger.debug(f"Density: {density:.1f} pts/m³, Adaptive factor: {adaptive_factor:.3f}, "
                    f"Radius: {base_radius:.3f} -> {adaptive_radius:.3f}")
        
        return adaptive_radius
    
    def compute_normals(self, points):
        adaptive_radius = self.compute_adaptive_radius(points, self.normal_radius)
        
        nbrs = NearestNeighbors(radius=adaptive_radius, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.radius_neighbors(points, radius=adaptive_radius)
        
        normals = np.zeros_like(points)
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if len(idx) < 3:
                normals[i] = [0, 0, 1]
                continue
                
            neighbors = points[idx]
            
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = np.cov(centered.T)
            
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            normal = eigenvecs[:, 0]
            
            if np.dot(normal, points[i]) < 0:
                normal = -normal
                
            normals[i] = normal
            
        return normals
    
    def compute_pca_features(self, points, normals):
        adaptive_radius = self.compute_adaptive_radius(points, self.eigen_radius)
        
        nbrs = NearestNeighbors(radius=adaptive_radius, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.radius_neighbors(points, radius=adaptive_radius)
        
        features = np.zeros((len(points), 7))
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if len(idx) < 3:
                features[i] = [0, 0, 0, 0, 0, 0, 0]
                continue
                
            neighbors = points[idx]
            
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = np.cov(centered.T)
            
            eigenvals = np.sort(np.linalg.eigvals(cov))[::-1]
            eigenvals = np.real(eigenvals)
            eigenvals = np.maximum(eigenvals, 1e-8)
            
            lambda1, lambda2, lambda3 = eigenvals
            
            linearity = (lambda1 - lambda2) / lambda1
            
            planarity = (lambda2 - lambda3) / lambda1
            
            sphericity = lambda3 / lambda1
            
            anisotropy = (lambda1 - lambda3) / lambda1
            
            omnivariance = np.power(lambda1 * lambda2 * lambda3, 1/3)
            
            eigenentropy = -np.sum(eigenvals * np.log(eigenvals + 1e-8))
            
            curvature = lambda3 / (lambda1 + lambda2 + lambda3)
            
            features[i] = [linearity, planarity, sphericity, anisotropy, 
                          omnivariance, eigenentropy, curvature]
        
        return features
    
    def compute_verticality(self, points, normals):
        cos_angles = np.abs(np.dot(normals, self.gravity))
        verticality = np.arccos(np.clip(cos_angles, -1, 1))
        
        return verticality
    
    def extract_features(self, points):

        normals = self.compute_normals(points)
        
        pca_features = self.compute_pca_features(points, normals)
        
        verticality = self.compute_verticality(points, normals)
        
        pca_mean = np.mean(pca_features, axis=0)
        pca_std = np.std(pca_features, axis=0)
        pca_pooled = np.concatenate([pca_mean, pca_std])
        
        vert_mean = np.mean(verticality)
        vert_std = np.std(verticality)
        vert_pooled = np.array([vert_mean, vert_std])
        
        geometry_features = np.concatenate([pca_pooled, vert_pooled])
        
        return geometry_features
