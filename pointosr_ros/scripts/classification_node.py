import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import numpy as np

class ClassifierNode:
    def __init__(self):
        rospy.init_node('classifier_node')

        # subscribe to cluster topics
        self.subs = []
        self.num_clusters = rospy.get_param('~num_clusters', 3)
        for i in range(self.num_clusters):
            topic = f'cluster_{i}'
            sub = rospy.Subscriber(topic, PointCloud2, self.callback, callback_args=i)
            self.subs.append(sub)

        self.buffer = {}
        self.frame_id = 0

        # publisher for classified output
        self.pub = rospy.Publisher('/classified/human', PointCloud2, queue_size=1)

    def callback(self, msg, cluster_id):
        rospy.loginfo(f"Received cluster {cluster_id}")
        # convert PointCloud2 to numpy array
        points = list(pc2.read_points(msg, skip_nans=True))
        self.buffer[cluster_id] = np.array(points)

        if len(self.buffer) == self.num_clusters:
            rospy.loginfo(f"All {self.num_clusters} clusters received for frame {self.frame_id}")

            # === call your deep learning model here ===
            predicted_classes = self.classify(self.buffer)

            # merge clusters by class
            merged_points = self.merge_clusters(predicted_classes, target_class="human")

            # publish merged cloud
            output_msg = pc2.create_cloud_xyz32(msg.header, merged_points)
            self.pub.publish(output_msg)

            # clear buffer for next frame
            self.buffer.clear()
            self.frame_id += 1

    def classify(self, cluster_dict):
        """
        Replace this stub with your actual classifier logic
        Return: dict {cluster_id: class_label}
        """
        result = {}
        for cid, points in cluster_dict.items():
            result[cid] = "human" if cid % 2 == 0 else "car"  # dummy classifier
        return result

    def merge_clusters(self, predicted_classes, target_class):
        merged = []
        for cid, class_label in predicted_classes.items():
            if class_label == target_class:
                merged.extend(self.buffer[cid])
        return merged

if __name__ == '__main__':
    node = ClassifierNode()
    rospy.spin()