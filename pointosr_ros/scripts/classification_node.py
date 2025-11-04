#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from pointosr_ros.msg import classification_batch, cluster_and_cls
import numpy as np
import torch
import threading
from collections import OrderedDict
import ros_numpy
import time
import json
import sys
import os

# Add the scripts directory to Python path for osr_scorer import
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pointnext.utils import EasyConfig, load_checkpoint
from pointnext.model import build_model_from_cfg
from pointnext.dataset.online.online_classification import OnlineDataloader
from osr_scorer import OSRScorer


class TimingStats:
    """Accumulator for timing statistics across batches."""
    def __init__(self):
        self.data_conversion_times = []
        self.model_inference_times = []
        self.osr_scoring_times = []
        self.result_packaging_times = []
        self.total_batch_times = []
        self.num_clusters_processed = 0
        self.num_batches = 0
    
    def add(self, data_conv, model, osr, packaging, total, num_clusters):
        """Add timing measurements for a batch."""
        self.data_conversion_times.append(data_conv)
        self.model_inference_times.append(model)
        self.osr_scoring_times.append(osr)
        self.result_packaging_times.append(packaging)
        self.total_batch_times.append(total)
        self.num_clusters_processed += num_clusters
        self.num_batches += 1
    
    def print_summary(self):
        """Print timing summary statistics."""
        if self.num_batches == 0:
            print("\n" + "="*60)
            print("PointOSR TIMING SUMMARY: No batches processed")
            print("="*60 + "\n")
            return
        
        # Convert to numpy arrays for statistics
        data_conv = np.array(self.data_conversion_times) * 1000  # Convert to ms
        model = np.array(self.model_inference_times) * 1000
        osr = np.array(self.osr_scoring_times) * 1000
        packaging = np.array(self.result_packaging_times) * 1000
        total = np.array(self.total_batch_times) * 1000
        
        # Calculate percentages
        mean_total = np.mean(total)
        pct_data = (np.mean(data_conv) / mean_total) * 100 if mean_total > 0 else 0
        pct_model = (np.mean(model) / mean_total) * 100 if mean_total > 0 else 0
        pct_osr = (np.mean(osr) / mean_total) * 100 if mean_total > 0 else 0
        pct_pkg = (np.mean(packaging) / mean_total) * 100 if mean_total > 0 else 0
        
        # Print summary
        print("\n" + "="*60)
        print("PointOSR TIMING SUMMARY")
        print("="*60)
        print(f"Total Batches: {self.num_batches}")
        print(f"Total Clusters Processed: {self.num_clusters_processed}")
        print(f"\nTiming Breakdown (mean ± stddev):")
        print(f"  Total Batch Time:   {np.mean(total):6.1f} ± {np.std(total):5.1f} ms  (100.0%)")
        print(f"  ├─ Data Conversion: {np.mean(data_conv):6.1f} ± {np.std(data_conv):5.1f} ms  ({pct_data:5.1f}%)")
        print(f"  ├─ Model Inference: {np.mean(model):6.1f} ± {np.std(model):5.1f} ms  ({pct_model:5.1f}%)")
        print(f"  ├─ OSR Scoring:     {np.mean(osr):6.1f} ± {np.std(osr):5.1f} ms  ({pct_osr:5.1f}%)")
        print(f"  └─ Result Packaging:{np.mean(packaging):6.1f} ± {np.std(packaging):5.1f} ms  ({pct_pkg:5.1f}%)")
        
        if self.num_clusters_processed > 0:
            per_cluster_time = (np.sum(total) / self.num_clusters_processed)
            print(f"\nPer-Cluster Average: {per_cluster_time:.2f} ms/cluster")
        
        print("="*60 + "\n")


class ClassificationNode:
    def __init__(self):
        rospy.init_node('pointosr_live_classification', anonymous=True)
        
        trigger_topic = rospy.get_param('~trigger_topic', '/motion_detector/cluster_batch')
        self.input_topic_prefix = rospy.get_param('~input_topic_prefix', '/cluster_')
        self.filtered_topic_prefix = rospy.get_param('~filtered_topic_prefix', '/filt_cluster_')
        max_cluster_topics = rospy.get_param('~max_cluster_topics', 30)
        self.buffer_timeout = rospy.get_param('~buffer_timeout', 10.0)

        model_path = rospy.get_param('~model_path')
        cfg_path = rospy.get_param('~cfg_path')
        device = rospy.get_param('~device', 'cuda')
        num_points = rospy.get_param('~num_points', 2048)
        
        # OSR parameters
        enable_osr = rospy.get_param('~enable_osr', True)
        fusion_config_path = rospy.get_param('~fusion_config_path', 'src/pointosr/calib_cache/fusion_config.json')
        stats_path = rospy.get_param('~stats_path', 'src/pointosr/calib_cache/stats.json')
        prototypes_path = rospy.get_param('~prototypes_path', 'src/pointosr/calib_cache/prototypes')

        self.start_time = time.time()
        self.initial_fusion_config_path = fusion_config_path
        self.initial_stats_path = stats_path
        self.initial_prototypes_path = prototypes_path
        self.calibration_stamp_path = None
        self.calibration_lock_path = None
        self.calibration_completed_at = None
        self.calibration_refresh_lock = threading.Lock()
        self.fusion_config_path = None
        self.stats_path = None
        self.prototypes_path = None
        
        # Initialize timing statistics
        self.timing_stats = TimingStats()
        rospy.on_shutdown(self._print_timing_summary)

        try:
            self.processor = OnlineDataloader(num_points=num_points, device=device)
            rospy.loginfo(f"Processor initialized for {num_points} points on '{device}'.")
            
            cfg = EasyConfig()
            cfg.load(cfg_path, recursive=True)
            cfg.update(rospy.get_param('~'))
            self.model = build_model_from_cfg(cfg.model).to(device)
            load_checkpoint(self.model, pretrained_path=model_path)
            self.model.eval()
            self.device = device
            rospy.loginfo(f"Model loaded from {model_path}.")
            self.class_names = cfg.get('classes', OnlineDataloader.classes)
            rospy.loginfo(f"Using classes: {self.class_names}")
            
            # Initialize OSR scorer with online calibration results
            self.enable_osr = enable_osr
            if self.enable_osr:
                try:
                    # Wait for calibration to complete
                    rospy.loginfo("⏳ Waiting for online calibration to complete...")
                    success, fusion_config_path, stats_path, prototypes_path, calibration_info = self._wait_for_calibration_completion(
                        fusion_config_path, stats_path, prototypes_path
                    )
                    if success:
                        rospy.loginfo("📁 Loading online calibration results...")
                        self.osr_scorer = OSRScorer(
                            fusion_config_path=fusion_config_path,
                            stats_path=stats_path,
                            prototypes_path=prototypes_path,
                            device=device
                        )
                        self.fusion_config_path = fusion_config_path
                        self.stats_path = stats_path
                        self.prototypes_path = prototypes_path
                        self.calibration_stamp_path = calibration_info.get('stamp_path') if calibration_info else None
                        self.calibration_lock_path = calibration_info.get('lock_path') if calibration_info else None
                        self.calibration_completed_at = calibration_info.get('completed_at') if calibration_info else None
                        # Setup embedding extraction for OSR
                        self._setup_embedding_extraction()
                        rospy.loginfo("✅ OSR (Open Set Recognition) enabled with online calibration")
                        rospy.loginfo(f"🎯 OOD threshold: {self.osr_scorer.ood_threshold:.6f}")
                        rospy.loginfo(f"🎯 Target TPR: {self.osr_scorer.target_tpr:.3f}")
                        rospy.loginfo(f"⚖️ Fusion weights: {self.osr_scorer.fusion_weights}")
                        # Warm up GPU
                        self._warmup_gpu()
                    else:
                        rospy.logwarn("⚠️ Online calibration failed or timed out!")
                        rospy.logwarn("🔄 Falling back to standard classification (no OSR)")
                        self.enable_osr = False
                        self.osr_scorer = None
                except Exception as e:
                    rospy.logerr(f"❌ Failed to initialize OSR scorer with online calibration: {e}")
                    rospy.logwarn("🔄 Falling back to standard classification (no OSR)")
                    self.enable_osr = False
                    self.osr_scorer = None
            else:
                self.osr_scorer = None
                rospy.loginfo("OSR disabled - using standard classification only")
                # Warm up GPU even without OSR
                self._warmup_gpu()
        except Exception as e:
            rospy.logerr(f"Failed to load model/config: {e}")
            rospy.signal_shutdown("Model loading failed.")
            return

        self.frame_buffer = OrderedDict()
        self.trigger_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()
        
        self.batch_publisher = rospy.Publisher(
            '/classified_clusters', classification_batch, queue_size=10
        )
        
        self.cluster_subscribers = [
            self._setup_subscriber(i) for i in range(max_cluster_topics)
        ]

        self.trigger_subscriber = rospy.Subscriber(trigger_topic, Header, self._trigger_callback)
        rospy.loginfo(f"Node ready. Listening for trigger on '{trigger_topic}'.")
        if self.enable_osr:
            rospy.loginfo("🎉 Online calibration successful - OSR enabled!")
        else:
            rospy.loginfo("ℹ️ Using standard classification (no OSR)")

    def _wait_for_calibration_completion(self, fusion_config_path, stats_path, prototypes_path, timeout=300):
        """Wait for calibration to complete by checking for required files.

        Returns:
            tuple: (success: bool, fusion_config_path: str, stats_path: str, prototypes_path: str)
        """
        # Get k_human and k_false parameters from ROS parameters
        k_human = rospy.get_param('~k_human', 8)  # Default to 8 if not set
        k_false = rospy.get_param('~k_false', 2)  # Default to 2 if not set
        
        # Resolve cache directory paths (same logic as calibration node)
        cache_dir = os.path.dirname(fusion_config_path)
        resolved_cache_dir = cache_dir
        if not os.path.isabs(cache_dir):
            # Find the pointosr repository root by traversing up from this file's location
            current_dir = os.path.dirname(os.path.abspath(__file__))  # This file is in pointosr_ros/scripts/
            pointosr_root = None

            # Traverse up the directory tree looking for a directory containing 'pointosr' subdirectory
            search_dir = current_dir
            max_depth = 10  # Prevent infinite loops
            depth = 0

            while search_dir != '/' and depth < max_depth:
                # Check if this directory contains a 'pointosr' subdirectory
                if os.path.exists(os.path.join(search_dir, 'pointosr')):
                    pointosr_root = search_dir
                    break
                search_dir = os.path.dirname(search_dir)
                depth += 1

            if pointosr_root:
                # Handle relative paths that start with 'src/pointosr/'
                if cache_dir.startswith('src/pointosr/'):
                    # Remove 'src/pointosr/' prefix and append to pointosr root
                    cache_subpath = cache_dir[13:]  # Remove 'src/pointosr/' prefix
                    resolved_cache_dir = os.path.join(pointosr_root, cache_subpath)
                else:
                    # Use the relative path as-is from pointosr root
                    resolved_cache_dir = os.path.join(pointosr_root, cache_dir)
                
                # Update all paths with resolved cache directory
                fusion_config_path = os.path.join(resolved_cache_dir, os.path.basename(fusion_config_path))
                stats_path = os.path.join(resolved_cache_dir, os.path.basename(stats_path))
                prototypes_path = os.path.join(resolved_cache_dir, os.path.basename(prototypes_path))
                
                rospy.loginfo(f"Resolved cache directory to: {resolved_cache_dir}")
            else:
                rospy.logwarn(f"Could not find pointosr repository root, using relative paths")
        
        stamp_path = os.path.join(os.path.dirname(fusion_config_path), 'calibration_complete.stamp')
        lock_path = f"{resolved_cache_dir}.lock"

        required_files = [
            fusion_config_path,
            stats_path,
            os.path.join(prototypes_path, f'human_k{k_human}', 'prototypes.npy'),
            os.path.join(prototypes_path, f'fp_k{k_false}', 'prototypes.npy'),
            stamp_path
        ]
        
        start_time = time.time()
        check_interval = 2.0  # Check every 2 seconds
        
        rospy.loginfo("🔍 Waiting for calibration files...")
        
        while time.time() - start_time < timeout:
            all_files_exist = True
            missing_files = []
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    all_files_exist = False
                    missing_files.append(file_path)
            
            if all_files_exist:
                # Validate K values in fusion_config match expected
                try:
                    with open(fusion_config_path, 'r') as f:
                        cfg = json.load(f)
                    cfg_kh = int(cfg.get('K_H', k_human))
                    cfg_kf = int(cfg.get('K_F', k_false))
                    if cfg_kh != k_human or cfg_kf != k_false:
                        rospy.logwarn(f"K mismatch: fusion_config has (K_H={cfg_kh}, K_F={cfg_kf}) but params are (k_human={k_human}, k_false={k_false}). Waiting for matching files...")
                        all_files_exist = False
                except Exception as e:
                    rospy.logwarn(f"Could not validate fusion_config K values: {e}")
                    all_files_exist = False

            if all_files_exist and os.path.exists(lock_path):
                all_files_exist = False
                missing_files.append(f"{lock_path} (calibration running)")

            completed_at = None
            if all_files_exist:
                try:
                    with open(stamp_path, 'r') as stamp_file:
                        stamp_data = json.load(stamp_file)
                    completed_at = float(stamp_data.get('completed_at', 0.0))
                    if completed_at < self.start_time:
                        all_files_exist = False
                        missing_files.append(f"{stamp_path} (stale results)")
                except Exception as e:
                    rospy.logwarn(f"Could not read calibration completion stamp: {e}")
                    all_files_exist = False

            if all_files_exist:
                rospy.loginfo("✅ All calibration files found!")
                calibration_info = {
                    'stamp_path': stamp_path,
                    'lock_path': lock_path,
                    'completed_at': completed_at
                }
                return True, fusion_config_path, stats_path, prototypes_path, calibration_info
            
            elapsed = time.time() - start_time
            rospy.loginfo(f"⏳ Waiting for calibration... ({elapsed:.1f}s elapsed) Missing: {len(missing_files)} items")
            for missing_file in missing_files:
                rospy.logdebug(f"Missing file: {missing_file}")
            
            # Check if we're still within timeout
            if elapsed >= timeout:
                rospy.logwarn(f"⏰ Timeout waiting for calibration ({timeout}s)")
                for missing_file in missing_files:
                    rospy.logwarn(f"Missing: {missing_file}")
                return False, fusion_config_path, stats_path, prototypes_path, None
            
            time.sleep(check_interval)
        
        return False, fusion_config_path, stats_path, prototypes_path, None

    def _calibration_needs_refresh(self):
        if not self.enable_osr:
            return False

        if self.osr_scorer is None:
            return True

        if self.calibration_lock_path and os.path.exists(self.calibration_lock_path):
            return True

        if not self.calibration_stamp_path:
            return False

        if not os.path.exists(self.calibration_stamp_path):
            return True

        try:
            with open(self.calibration_stamp_path, 'r') as stamp_file:
                stamp_data = json.load(stamp_file)
            completed_at = float(stamp_data.get('completed_at', 0.0))
        except Exception:
            return True

        if self.calibration_completed_at is None:
            return True

        return completed_at != self.calibration_completed_at

    def _reload_osr_calibration(self):
        success, fusion_config_path, stats_path, prototypes_path, calibration_info = self._wait_for_calibration_completion(
            self.initial_fusion_config_path,
            self.initial_stats_path,
            self.initial_prototypes_path
        )

        if not success or calibration_info is None:
            return False

        self.osr_scorer = OSRScorer(
            fusion_config_path=fusion_config_path,
            stats_path=stats_path,
            prototypes_path=prototypes_path,
            device=self.device
        )
        self.fusion_config_path = fusion_config_path
        self.stats_path = stats_path
        self.prototypes_path = prototypes_path
        self.calibration_stamp_path = calibration_info.get('stamp_path')
        self.calibration_lock_path = calibration_info.get('lock_path')
        self.calibration_completed_at = calibration_info.get('completed_at')
        rospy.loginfo("✅ OSR calibration reloaded with latest results")
        return True

    def _ensure_latest_calibration(self):
        if not self._calibration_needs_refresh():
            return True

        rospy.logwarn("⚠️ Calibration update detected — pausing until new results are ready")

        with self.calibration_refresh_lock:
            if not self._calibration_needs_refresh():
                return True

            success = self._reload_osr_calibration()
            if not success:
                rospy.logwarn("⚠️ Unable to refresh calibration — continuing without OSR")
                self.osr_scorer = None
                self.calibration_completed_at = None
                return False
            return True

    def _setup_embedding_extraction(self):
        """Setup model to extract both logits and embeddings for OSR."""
        # Store reference to current embeddings
        self.current_embeddings = None
        
        def embedding_hook(module, input, output):
            self.current_embeddings = output.detach()
        
        # Register hook on encoder
        if hasattr(self.model, 'encoder'):
            self.model.encoder.register_forward_hook(embedding_hook)
        else:
            rospy.logwarn("Model doesn't have 'encoder' attribute for OSR embedding extraction")
            
        # Override forward to ensure embeddings are captured
        original_forward = self.model.forward
        
        def enhanced_forward(data):
            if hasattr(self.model, 'encoder'):
                global_feat = self.model.encoder.forward_cls_feat(data)
                self.current_embeddings = global_feat
                logits = self.model.prediction(global_feat)
                return logits, global_feat
            else:
                return original_forward(data)
        
        self.model.forward = enhanced_forward
    
    def _warmup_gpu(self, num_warmup_runs=10):
        """Perform warm-up inference to initialize GPU kernels and memory."""
        rospy.loginfo("🔥 Warming up GPU with dummy inference...")
        
        # Create dummy data matching expected input shape
        dummy_pos = torch.randn(4, 2048, 3).to(self.device)  # 4 samples, 2048 points, 3D coords
        dummy_x = torch.randn(4, 4, 2048).to(self.device)    # 4 samples, 4 features, 2048 points
        dummy_data = {'pos': dummy_pos, 'x': dummy_x}
        
        with torch.no_grad():
            for i in range(num_warmup_runs):
                if self.enable_osr and self.osr_scorer is not None:
                    # Warm up both model and OSR
                    logits, embeddings = self.model(dummy_data)
                    try:
                        _ = self.osr_scorer.score_batch(logits, embeddings)
                    except Exception:
                        pass  # Ignore warmup errors
                else:
                    # Warm up model only
                    _ = self.model(dummy_data)
        
        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()
        rospy.loginfo(f"✅ GPU warm-up complete ({num_warmup_runs} runs)")

    def _setup_subscriber(self, index):
        topic_name = f"{self.input_topic_prefix}{index}"
        return rospy.Subscriber(
            topic_name, PointCloud2,
            lambda msg, tn=topic_name: self._cluster_callback(msg, tn)
        )

    def _cluster_callback(self, msg, topic_name):
        with self.buffer_lock:
            stamp = msg.header.stamp
            now = rospy.Time.now()
            
            if stamp not in self.frame_buffer:
                self.frame_buffer[stamp] = {'messages': {}, 'arrival_time': now}
            
            self.frame_buffer[stamp]['messages'][topic_name] = msg
            
            if stamp in self.trigger_buffer:
                expected_count = self.trigger_buffer[stamp]['expected_count']
                if len(self.frame_buffer[stamp]['messages']) == expected_count:
                    self._process_frame_if_complete(stamp)

    def _trigger_callback(self, header_msg):
        with self.buffer_lock:
            stamp = header_msg.stamp
            expected_count = header_msg.seq
            now = rospy.Time.now()

            if expected_count == 0:
                rospy.logdebug(f"Trigger for stamp {stamp} received with 0 expected clusters — publishing empty batch.")
                self.buffer_lock.release()
                try:
                    self._process_batch([], [], stamp, 0)
                finally:
                    self.buffer_lock.acquire()
                return

            self.trigger_buffer[stamp] = {'expected_count': expected_count, 'arrival_time': now}

            if stamp in self.frame_buffer:
                if len(self.frame_buffer[stamp]['messages']) == expected_count:
                    self._process_frame_if_complete(stamp)

            self._cleanup_buffers()

    def _process_frame_if_complete(self, stamp):
        rospy.logdebug(f"Frame for stamp {stamp} is complete. Processing.")
        
        batch_data_dict = self.frame_buffer.pop(stamp)['messages']
        expected_count = self.trigger_buffer.pop(stamp)['expected_count']
        
        self.buffer_lock.release()
        
        try:
            topic_list = list(batch_data_dict.keys())
            msgs = [batch_data_dict[tn] for tn in topic_list]
            
            rospy.logdebug(f"Processing batch of {len(msgs)} clusters for stamp {stamp}.")
            self._process_batch(msgs, topic_list, stamp, expected_count)
        finally:
            self.buffer_lock.acquire()

    def _cleanup_buffers(self):
        now = rospy.Time.now()
        
        for s, data in list(self.frame_buffer.items()):
            if (now - data['arrival_time']).to_sec() > self.buffer_timeout:
                expected = self.trigger_buffer.get(s, {}).get('expected_count', 'N/A')
                rospy.logwarn(f"Timing out stale frame {s} (recv: {len(data['messages'])}, expect: {expected}).")
                del self.frame_buffer[s]
                if s in self.trigger_buffer:
                    del self.trigger_buffer[s]
        
        for s, data in list(self.trigger_buffer.items()):
            if (now - data['arrival_time']).to_sec() > self.buffer_timeout:
                rospy.logwarn(f"Timing out stale trigger {s} (expect: {data['expected_count']}, recv: 0).")
                del self.trigger_buffer[s]

    def _process_batch(self, msgs, topic_list, stamp, expected_count):
        start_time = time.time()
        
        # Initialize timing accumulators
        data_conversion_time = 0.0
        model_inference_time = 0.0
        osr_scoring_time = 0.0
        result_packaging_time = 0.0
        
        batch_result = classification_batch()
        batch_result.header.stamp = stamp
        batch_result.header.frame_id = "classification_batch"
        batch_result.total_input_clusters = expected_count
        batch_result.total_processed_clusters = 0
        batch_result.human_count = 0
        batch_result.fp_count = 0
        batch_result.ood_count = 0
        batch_result.processing_errors = 0
        batch_result.processing_log = []
        batch_result.classified_clusters = []
        
        if not msgs:
            batch_result.processing_log.append(f"No messages received for stamp {stamp}")
            self._publish_batch_result(batch_result, start_time, 
                                      data_conversion_time, model_inference_time, 
                                      osr_scoring_time, result_packaging_time)
            return

        use_osr = False
        if self.enable_osr:
            osr_ready = self._ensure_latest_calibration()
            if osr_ready and self.osr_scorer is not None:
                use_osr = True
            else:
                if not osr_ready:
                    rospy.logwarn("OSR calibration not ready — running standard classification for this batch")
                else:
                    rospy.logwarn("OSR scorer unavailable — running standard classification for this batch")

        batch_pos, batch_x, valid_indices, valid_msgs = [], [], [], []
        
        # Start timing data conversion
        t_data_start = time.time()
        
        try:
            for i, msg in enumerate(msgs):
                topic_name = topic_list[i]
                
                try:
                    cluster_index = int(topic_name.split('_')[-1])
                except (ValueError, IndexError):
                    cluster_index = i
                
                try:
                    pc_data = ros_numpy.numpify(msg)
                    points = np.zeros((len(pc_data), 4), dtype=np.float32)
                    points[:, 0], points[:, 1], points[:, 2], points[:, 3] = pc_data['x'], pc_data['y'], pc_data['z'], pc_data['intensity']
                    points = points[np.isfinite(points).all(axis=1)]

                    if points.shape[0] == 0:
                        failed_cluster = self._create_failed_cluster_result(msg, cluster_index, "Empty point cloud")
                        batch_result.classified_clusters.append(failed_cluster)
                        batch_result.processing_errors += 1
                        batch_result.processing_log.append(f"Cluster {cluster_index}: Empty point cloud")
                        continue

                    centroid = Point()
                    centroid.x = float(np.mean(points[:, 0]))
                    centroid.y = float(np.mean(points[:, 1]))
                    centroid.z = float(np.mean(points[:, 2]))

                    data = self.processor.process(points)
                    batch_pos.append(data['pos'])
                    batch_x.append(data['x'])
                    valid_indices.append(cluster_index)
                    valid_msgs.append((msg, centroid, len(points)))
                    
                except Exception as e:
                    failed_cluster = self._create_failed_cluster_result(msg, cluster_index, str(e))
                    batch_result.classified_clusters.append(failed_cluster)
                    batch_result.processing_errors += 1
                    batch_result.processing_log.append(f"Cluster {cluster_index}: Processing error - {str(e)}")

            batch_result.total_processed_clusters = len(valid_msgs)
            
            # End timing data conversion
            data_conversion_time = time.time() - t_data_start

            if not batch_pos:
                batch_result.processing_log.append(f"All {len(msgs)} clusters failed preprocessing")
                self._publish_batch_result(batch_result, start_time,
                                          data_conversion_time, model_inference_time,
                                          osr_scoring_time, result_packaging_time)
                return 

            pos_tensor = torch.stack(batch_pos, dim=0).to(self.device)
            x_tensor = torch.stack(batch_x, dim=0).to(self.device)
            model_data = {'pos': pos_tensor, 'x': x_tensor.transpose(1, 2)}
            
            # Start timing model inference
            t_model_start = time.time()
            
            with torch.no_grad():
                if use_osr:
                    # OSR-enabled forward pass - get both logits and embeddings
                    logits, embeddings = self.model(model_data)
                    pred_indices = torch.argmax(logits, dim=1)
                    confidences = torch.softmax(logits, dim=1)
                else:
                    # Standard forward pass - get only logits
                    logits = self.model(model_data)
                    pred_indices = torch.argmax(logits, dim=1)
                    confidences = torch.softmax(logits, dim=1)
            
            # End timing model inference
            model_inference_time = time.time() - t_model_start
            
            # Start timing OSR scoring (if enabled)
            t_osr_start = time.time()
            
            with torch.no_grad():
                if use_osr:
                    # Perform OSR scoring
                    try:
                        osr_results = self.osr_scorer.score_batch(logits, embeddings)
                        if osr_results is None or 'energy_scores_raw' not in osr_results:
                            rospy.logwarn("OSR scoring failed - falling back to standard classification")
                            osr_results = None
                    except Exception as e:
                        rospy.logerr(f"OSR scoring error: {e}")
                        osr_results = None
                else:
                    osr_results = None
            
            # End timing OSR scoring
            osr_scoring_time = time.time() - t_osr_start
            
            # Start timing result packaging
            t_package_start = time.time()
            
            with torch.no_grad():
                for i in range(pred_indices.shape[0]):
                    class_idx = pred_indices[i].item()
                    confidence = float(confidences[i, class_idx])
                    cluster_index = valid_indices[i]
                    msg, centroid, cluster_size = valid_msgs[i]
                    
                    # Determine final class name and OOD status
                    if use_osr and osr_results is not None:
                        is_ood = osr_results['is_ood'][i]
                        class_name = self.osr_scorer.get_class_name_with_ood(class_idx, is_ood, self.class_names)
                        
                        # OSR scores
                        energy_score = float(osr_results['energy_scores_raw'][i])
                        cosine_score = float(osr_results['cosine_scores_raw'][i])
                        fused_score = float(osr_results['fused_scores'][i])
                        ood_confidence = float(osr_results['ood_confidences'][i])
                    else:
                        # Standard classification without OSR
                        class_name = self.class_names[class_idx]
                        is_ood = False
                        energy_score = 0.0
                        cosine_score = 0.0
                        fused_score = 0.0
                        ood_confidence = 0.0
                    
                    cluster_result = cluster_and_cls()
                    cluster_result.header = msg.header
                    cluster_result.header.stamp = stamp
                    cluster_result.pointcloud = msg
                    cluster_result.class_name = class_name
                    cluster_result.confidence = confidence
                    cluster_result.is_human = (class_name.lower() == "human" or class_name.lower() == "ood")
                    
                    # OSR fields
                    cluster_result.is_ood = is_ood
                    cluster_result.energy_score = energy_score
                    cluster_result.cosine_score = cosine_score
                    cluster_result.fused_score = fused_score
                    cluster_result.ood_confidence = ood_confidence
                    
                    cluster_result.original_cluster_index = cluster_index
                    cluster_result.processing_success = True
                    cluster_result.error_message = ""
                    cluster_result.centroid = centroid
                    cluster_result.cluster_size = cluster_size
                    
                    batch_result.classified_clusters.append(cluster_result)
                    
                    # Update counters
                    if class_name.lower() == "human":
                        batch_result.human_count += 1
                    elif class_name.lower() == "ood":
                        batch_result.ood_count += 1
                    else:
                        batch_result.fp_count += 1
                    
                    # Logging
                    if use_osr and is_ood:
                        batch_result.processing_log.append(
                            f"Cluster {cluster_index}: '{class_name}' (conf: {confidence:.3f}, "
                            f"fused: {fused_score:.3f}, ood_conf: {ood_confidence:.3f})"
                        )
                    else:
                        batch_result.processing_log.append(f"Cluster {cluster_index}: '{class_name}' (conf: {confidence:.3f})")
            
            # End timing result packaging
            result_packaging_time = time.time() - t_package_start

            self._publish_batch_result(batch_result, start_time,
                                      data_conversion_time, model_inference_time,
                                      osr_scoring_time, result_packaging_time)

        except Exception as e:
            batch_result.processing_log.append(f"Critical error in batch processing: {str(e)}")
            batch_result.processing_errors += 1
            rospy.logerr(f"Error processing batch for stamp {stamp}: {e}")
            self._publish_batch_result(batch_result, start_time,
                                      data_conversion_time, model_inference_time,
                                      osr_scoring_time, result_packaging_time)

    def _create_failed_cluster_result(self, msg, cluster_index, error_message):
        failed_cluster = cluster_and_cls()
        failed_cluster.header = msg.header
        failed_cluster.pointcloud = msg
        failed_cluster.class_name = "processing_failed"
        failed_cluster.confidence = 0.0
        failed_cluster.is_human = False
        
        # OSR fields for failed clusters
        failed_cluster.is_ood = False
        failed_cluster.energy_score = 0.0
        failed_cluster.cosine_score = 0.0
        failed_cluster.fused_score = 0.0
        failed_cluster.ood_confidence = 0.0
        
        failed_cluster.original_cluster_index = cluster_index
        failed_cluster.processing_success = False
        failed_cluster.error_message = error_message
        failed_cluster.centroid = Point()
        failed_cluster.cluster_size = 0
        return failed_cluster
    
    def _publish_batch_result(self, batch_result, start_time, 
                             data_conversion_time, model_inference_time,
                             osr_scoring_time, result_packaging_time):
        try:
            batch_result.processing_time_sec = time.time() - start_time
            batch_result.model_version = "pointnext-s"
            
            # Add timing to accumulator
            self.timing_stats.add(data_conversion_time, model_inference_time,
                                 osr_scoring_time, result_packaging_time,
                                 batch_result.processing_time_sec,
                                 batch_result.total_processed_clusters)
            
            self.batch_publisher.publish(batch_result)
            
            total_clusters = batch_result.total_input_clusters
            processed_clusters = batch_result.total_processed_clusters
            human_count = batch_result.human_count
            fp_count = batch_result.fp_count
            ood_count = batch_result.ood_count
            error_count = batch_result.processing_errors
            
            # Convert to milliseconds for logging
            total_ms = batch_result.processing_time_sec * 1000
            data_ms = data_conversion_time * 1000
            model_ms = model_inference_time * 1000
            osr_ms = osr_scoring_time * 1000
            pkg_ms = result_packaging_time * 1000
            
            if self.enable_osr and self.osr_scorer is not None:
                rospy.loginfo(f"Batch @ {batch_result.header.stamp.to_sec():.3f}: "
                             f"{total_ms:.1f}ms (Data={data_ms:.1f}ms, Model={model_ms:.1f}ms, "
                             f"OSR={osr_ms:.1f}ms, Pkg={pkg_ms:.1f}ms) | "
                             f"Clusters: {processed_clusters}/{total_clusters}, "
                             f"Human={human_count}, FP={fp_count}, OOD={ood_count}, Errors={error_count}")
            else:
                rospy.loginfo(f"Batch @ {batch_result.header.stamp.to_sec():.3f}: "
                             f"{total_ms:.1f}ms (Data={data_ms:.1f}ms, Model={model_ms:.1f}ms, "
                             f"Pkg={pkg_ms:.1f}ms) | "
                             f"Clusters: {processed_clusters}/{total_clusters}, "
                             f"Human={human_count}, FP={fp_count}, Errors={error_count}")
            
            if rospy.get_param('~debug_logging', False):
                for log_entry in batch_result.processing_log:
                    rospy.logdebug(log_entry)
                    
        except Exception as e:
            rospy.logerr(f"Error publishing batch result: {e}")
    
    def _print_timing_summary(self):
        """Print timing summary on shutdown."""
        self.timing_stats.print_summary()

if __name__ == '__main__':
    try:
        ClassificationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in node initialization: {e}")