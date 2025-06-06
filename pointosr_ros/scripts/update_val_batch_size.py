import rospy
import yaml
import sys
import os
import time

def count_cluster_topics():
    """Counts the number of topics starting with '/cluster_'."""
    # Using print for output as this is a command-line script.
    try:
        # log_level=rospy.ERROR will suppress INFO messages from rospy on init.
        rospy.init_node('cluster_counter', anonymous=True, log_level=rospy.ERROR)
        
        print("Counting cluster topics...")
        # A short delay can help ensure all topics are discovered after connecting to master.
        time.sleep(1.0)
        topics = rospy.get_published_topics()
        cluster_count = sum(1 for topic, _ in topics if topic.startswith('/cluster_'))
        
        # If no clusters are found, wait a bit longer and retry once.
        if cluster_count == 0:
            print("No cluster topics found on first try, waiting and retrying...")
            time.sleep(2.0)
            topics = rospy.get_published_topics()
            cluster_count = sum(1 for topic, _ in topics if topic.startswith('/cluster_'))

        if cluster_count == 0:
            print("Final check: No cluster topics found.")
            return 0
            
        print(f"Found {cluster_count} cluster topics.")
        return cluster_count
    except rospy.ROSInterruptException:
        print("Error: ROS interrupt received while counting topics.")
        return -1
    except Exception as e:
        print(f"An error occurred while counting clusters: {e}.")
        return -1

def update_yaml_file(file_path, key, value):
    """Updates a key in a YAML file."""
    if not os.path.isfile(file_path):
        print(f"Error: YAML file not found at: {file_path}")
        return False

    try:
        with open(file_path, 'r') as f:
            # Use safe_load to avoid security vulnerabilities.
            config = yaml.safe_load(f)

        if key in config and config[key] == value:
            print(f"'{key}' is already set to '{value}' in '{file_path}'. No update needed.")
            return True

        config[key] = value

        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Successfully updated '{key}' to '{value}' in '{file_path}'")
        return True
    except Exception as e:
        print(f"Error: Failed to update YAML file '{file_path}': {e}")
        return False


if __name__ == "__main__":
    # Check for PyYAML dependency first.
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is not installed. Please install it using: pip install pyyaml")
        sys.exit(1)
        
    if len(sys.argv) < 2:
        print("Usage: python update_val_batch_size.py <path_to_config.yaml>")
        sys.exit(1)
    
    config_file_path = sys.argv[1]

    # Check if a ROS master is running before attempting to use rospy.
    try:
        rospy.get_master().getSystemState()
    except Exception as e:
        print(f"Error: Could not connect to ROS master. Please make sure roscore and your nodes are running.")
        print(f"Details: {e}")
        sys.exit(1)

    cluster_count = count_cluster_topics()
    if cluster_count > 0:
        if not update_yaml_file(config_file_path, 'val_batch_size', cluster_count):
            sys.exit(1)
    elif cluster_count == 0:
        print("No cluster topics found. Inference will be skipped.")
        sys.exit(2)
    else: # cluster_count < 0
        print("An error occurred during cluster counting. Exiting.")
        sys.exit(1) 