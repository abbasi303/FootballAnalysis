Football Analysis Project

Overview

This project leverages computer vision techniques to analyze football games. It uses the YOLO object detection model and additional clustering and tracking algorithms to detect and differentiate players, referees, and the ball. The project also involves segmenting players by their team based on shirt colors, analyzing ball and player movements, and tracking camera movements to gain insights into game dynamics.

Workflow

1. Object Detection with YOLO

Model: YOLO (You Only Look Once) is used for detecting objects in input football videos.

Classes Displayed: The processed video highlights detected classes, including players, referees, and the football.

2. Dataset Preparation and Model Optimization

Dataset: Roboflow’s football dataset is used to fine-tune YOLO to improve detection accuracy.

Optimization:

Address issues where players outside the field are detected.

Differentiate between players, referees, and ball.

Train the YOLO model using the Roboflow dataset in Kaggle.

Save the best and last YOLO model weights for further use.

3. Folder Organization

The football_detection folder is duplicated and moved to a new directory for backup and separate experimentation.

YOLO training scripts (football_training) are run on Kaggle.

4. Custom Enhancements

Bounding Box to Circle: Bounding boxes for detected objects are changed to circular shapes for better visualization.

Ball Tracking: A triangle tracker is added specifically for tracking the ball's movement.

5. Player Segmentation and Analysis

Team Differentiation: Players are segmented and grouped based on shirt color using:

Image cropping for isolated player images.

Clustering techniques to analyze and isolate team colors from pixel data.

Tools:

Roboflow for dataset preparation.

Libraries like Scikit-learn, Pandas, NumPy for analysis.

6. Ball Interpolation

Ball positions are interpolated to ensure smooth tracking and movement analysis.

7. Camera Movement Analysis

Importance: Understanding camera movements helps in analyzing the speed and movement of objects in the video.

Techniques: Perspective transformation is applied to correct camera angles and enhance the accuracy of object tracking.

Results and Insights

Accurate detection and differentiation of players, referees, and ball.

Clear segmentation of players by team.

Enhanced ball tracking with smooth interpolation.

Improved understanding of game dynamics through camera movement analysis.

Folder Structure

Input Video: Raw footage of football games.

Training Folder:

Contains scripts for YOLO training.

Includes model weights (best.pt, last.pt).

Detection Folder:

Stores processed videos with detected objects.

Circular bounding boxes and ball tracking results.

Segmentation Folder:

Cropped player images.

Clustered images based on shirt colors.

Analysis Folder:

Outputs from movement and perspective transformation analysis.

Libraries and Tools

YOLO (Ultralytics)

Roboflow (Dataset Preparation)

Pandas, NumPy (Data Manipulation)

Scikit-learn (Clustering)

Supervision (Tracking)

Steps to Run the Project

Prepare the dataset using Roboflow and upload it to Kaggle.

Train the YOLO model by running the football_training script.

Download the best and last weights.

Process the input video using YOLO with the trained weights.

Customize bounding boxes and add ball tracking.

Perform player segmentation and clustering.

Conduct movement analysis and perspective transformation.

Future Improvements

Implement advanced tracking algorithms for better player movement analysis.

Incorporate deeper segmentation techniques for more accurate team differentiation.

Expand analysis to include player performance metrics like speed, distance covered, and goal probability.
