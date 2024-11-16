from ultralytics import YOLO
import supervision as sv
import json
import pickle
import os
import cv2
import sys
sys.path.append('../')
from utils import get_bbox_width,get_center_of_bbox
import random
import numpy as np
import pandas as pd
from pykalman import KalmanFilter


class Tracker:
    def __init__(self, model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()

    def smooth_ball_positions(self, ball_positions):
        # Extract ball center positions
        centers = [
            ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) if bbox else (None, None)
            for bbox in ball_positions
        ]

        # Replace None values with the last valid position for Kalman filter input
        valid_centers = []
        last_valid = None
        for center in centers:
            if center[0] is not None:
                valid_centers.append(center)
                last_valid = center
            else:
                # If no valid position, use the last known valid position
                valid_centers.append(last_valid if last_valid else (0, 0))  # Default to (0, 0) if no valid position yet

        valid_centers = np.array(valid_centers)

        # Adaptive Kalman filter parameters
        transition_covariance = 1e-4 * np.eye(4)  # Smaller values for faster response
        observation_covariance = 1e-1 * np.eye(2)  # Trust observations more

        # Create Kalman filter
        kf = KalmanFilter(
            transition_matrices=[[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            observation_matrices=[[1, 0, 0, 0], [0, 1, 0, 0]],
            initial_state_mean=[valid_centers[0][0], valid_centers[0][1], 0, 0],
            initial_state_covariance=1e-3 * np.eye(4),
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )

        # Train Kalman filter with valid observations
        kf = kf.em(valid_centers, n_iter=10)

        # Perform smoothing
        smoothed_positions, _ = kf.smooth(valid_centers)

        # Convert smoothed positions to bounding boxes
        smoothed_ball_positions = []
        for idx, center in enumerate(centers):
            if center[0] is None:
                # Use the smoothed position for missing frames
                smoothed_center = smoothed_positions[idx]
                smoothed_ball_positions.append([
                    smoothed_center[0] - 10,  # Adjust bbox width
                    smoothed_center[1] - 10,  # Adjust bbox height
                    smoothed_center[0] + 10,
                    smoothed_center[1] + 10
                ])
            else:
                # For high-confidence detections, bypass Kalman filter
                velocity = (
                    np.linalg.norm(
                        np.array(center) - np.array(smoothed_positions[idx - 1, :2])
                    )
                    if idx > 0
                    else 0
                )
                if velocity > 75:  # Threshold for rapid updates
                    smoothed_ball_positions.append([
                        center[0] - 10,
                        center[1] - 10,
                        center[0] + 10,
                        center[1] + 10
                    ])
                else:
                    # Use the smoothed position
                    smoothed_ball_positions.append([
                        smoothed_positions[idx, 0] - 10,
                        smoothed_positions[idx, 1] - 10,
                        smoothed_positions[idx, 0] + 10,
                        smoothed_positions[idx, 1] + 10
                    ])

        return smoothed_ball_positions



    
    def interpolate_ball_positions(self, ball_positions):
        # Extract bounding boxes
        extracted_positions = [
            list(frame_data.values())[0].get('bbox', []) if frame_data else []
            for frame_data in ball_positions
        ]

        # print("Extracted ball positions before smoothing:", extracted_positions)

        # Apply Kalman filter with prediction
        smoothed_positions = self.smooth_ball_positions(extracted_positions)

        # Rebuild ball positions in the expected format
        ball_positions = [
            {1: {"bbox": bbox}} if bbox else {} for bbox in smoothed_positions
        ]

        return ball_positions


    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0, len(frames),batch_size):
            detections_batch= self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks





    def draw_ellipse(self, frame, bbox, color, track_id=None):
        overlay = frame.copy()
        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        height = int(0.35 * width)
        # Draw circle instead of ellipse for testing
        cv2.ellipse(
                    frame,
                    center=(x_center,y_center),
                    axes=(int(width), height),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color = color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
        
        alpha = 0.1  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y_center- rectangle_height//2) +15
        y2_rect = (y_center+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        # print(f"Ellipse at: ({x_center}, {y_center}), Width: {width}, Height: {height}")  # Debugging
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        # Check if bbox is valid (no NaN values)
        if any(np.isnan(coord) for coord in bbox):
            print("Skipping drawing triangle due to invalid bbox:", bbox)
            return frame

        # Extract coordinates
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Define triangle points
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ], np.int32)

        # Draw the triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame


    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    #     # Optional: draw a dot at the center for visual confirmation
    #     cv2.circle(frame, (x_center, y_center), 5, (0, 255, 255), -1)  # Yellow dot
    #     return frame

        
    def draw_annotations(self, video_frames, tracks,team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # Make a copy of the frame to avoid modifying the original

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw ellipses on players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw ellipses on referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))

            # Optionally, draw ellipses on the ball as well
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

