import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)

class FPFHFeatureExtractor:
    
    def __init__(self, config):
        self.fpfh_radius = config.geometry.fpfh_radii[0]
        self.pca_k = config.geometry.fpfh_pca_k
        self.min_neighbors = config.geometry.min_neighbors
        self.pca_transformer = None
        
        self.density_adaptive = getattr(config.geometry, 'density_adaptive', True)
        self.base_density = getattr(config.geometry, 'base_density', 1000)
        self.min_radius = getattr(config.geometry, 'min_radius', 0.05)
        self.max_radius = getattr(config.geometry, 'max_radius', 0.5)
        
        logger.info(f"FPFHFeatureExtractor initialized:")
        logger.info(f"  FPFH radius: {self.fpfh_radius}")
        logger.info(f"  PCA components: {self.pca_k}")
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
        
        logger.debug(f"FPFH - Density: {density:.1f} pts/m³, Adaptive factor: {adaptive_factor:.3f}, "
                    f"Radius: {base_radius:.3f} -> {adaptive_radius:.3f}")
        
        return adaptive_radius
    
    def compute_fpfh(self, points):
        adaptive_radius = self.compute_adaptive_radius(points, self.fpfh_radius)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=adaptive_radius)
        )
        
        search_param = o3d.geometry.KDTreeSearchParamRadius(radius=adaptive_radius)
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)
        
        fpfh_features = np.asarray(fpfh.data).T
        
        return fpfh_features
    
    def fit_pca(self, train_fpfh_features):
        logger.info(f"Fitting PCA on {train_fpfh_features.shape[0]} training samples...")
        
        self.pca_transformer = PCA(n_components=self.pca_k)
        self.pca_transformer.fit(train_fpfh_features)
        
        explained_variance = np.sum(self.pca_transformer.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_variance:.3f}")
        logger.info(f"PCA components: {self.pca_transformer.components_.shape}")
    
    def transform_fpfh(self, fpfh_features):

        if self.pca_transformer is None:
            raise ValueError("PCA transformer not fitted. Call fit_pca() first.")
        
        transformed = self.pca_transformer.transform(fpfh_features)
        return transformed
    
    def extract_features(self, points):

        fpfh_features = self.compute_fpfh(points)
        
        if self.pca_transformer is not None:
            fpfh_transformed = self.transform_fpfh(fpfh_features)
        else:
            fpfh_transformed = fpfh_features[:, :6]
        
        fpfh_mean = np.mean(fpfh_transformed, axis=0)
        fpfh_std = np.std(fpfh_transformed, axis=0)
        fpfh_pooled = np.concatenate([fpfh_mean, fpfh_std])
        
        return fpfh_pooled
    
    def save_pca(self, filepath):
        if self.pca_transformer is not None:
            import joblib
            joblib.dump(self.pca_transformer, filepath)
            logger.info(f"PCA transformer saved to {filepath}")
    
    def load_pca(self, filepath):
        import joblib
        self.pca_transformer = joblib.load(filepath)
        logger.info(f"PCA transformer loaded from {filepath}")
