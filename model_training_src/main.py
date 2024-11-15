import os
from ultralytics import YOLO
import yaml

# Collect the base path dynamically
base_path = os.path.dirname(os.path.abspath(__file__))

# Define your data dictionary with dynamic paths
data_dict = {
    'path': os.path.join(base_path, 'data'),
    'train': os.path.join(base_path, 'data', 'images', 'train'),
    'val': os.path.join(base_path, 'data', 'images', 'train'),
    'names': {
        0: 'battery_with_label',
        1: 'battery_no_label'
    }
}

# Save data to a temporary YAML file
with open('temp_data.yaml', 'w') as outfile:
    yaml.dump(data_dict, outfile, default_flow_style=False)

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Train the model
model.train(data="temp_data.yaml", epochs=10)  # Use the temporary YAML file
