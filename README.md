Applied Vision AI
This repository contains a series of example programs to support vision AI applications for practical industry deployment. The programs are built using the Ultralytics YOLO v8 library.

Instructions
1. Clone the GitHub Repository
bash
git clone https://github.com/PeterDCodes/applied-visionAI.git
cd applied-visionAI

2. Create and Activate a Virtual Environment for Python
bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

3. Install the Required Libraries
    bash
    pip install -r requirements.txt

4. Repository Contents
    •	src/: Contains a series of programs you can use to demo the ultralytics functions. All programs return an output video file.
    •	model.pt: A pre-trained model created to detect a 'clock' object shown in the videos.
    •	video_1 and video_2: Example videos to use with the programs.
    •	model_training_src/: Contains a demonstration of how to train your own model.
    
5. Training Your Own Model
    In the model_training_src directory, you will find scripts that demonstrate how to train your own model.
    •	main.py: This script sets up the training environment by dynamically defining the paths to your training and validation data using the os module. It then builds a new YOLO model from scratch and trains it using the specified data.
    •	temp_data.yaml: This temporary YAML file is created by the main.py script to store the dataset configuration in a format that the YOLO model can use for training.
    How the Training Script Works:
    1.	Dynamic Path Collection: The script uses the os module to dynamically collect the base path, ensuring that the paths are user-agnostic and work on any system.
    2.	Data Dictionary: It defines a data dictionary with paths to the training and validation datasets.
    3.	Temporary YAML File: The data dictionary is saved into a temporary YAML file (temp_data.yaml).
    4.	Model Training: The script then loads a YOLO model configuration file (yolov8n.yaml) and trains the model using the data specified in temp_data.yaml.

