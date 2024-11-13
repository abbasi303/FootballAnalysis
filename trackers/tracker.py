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

class Tracker:
    def __init__(self, model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()
        self.track_colors = {}


    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0, len(frames),batch_size):
            detections_batch= self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections


    def get_object_tracks(self, frames, read_from_stub,stub_path):

        if read_from_stub is True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks
        
        detections =self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]

        }
        for frame_num , detections in enumerate(detections):
            cls_names =detections.names
            cls_names_inv= {v:k for k,v in cls_names.items()}

            # Convert to supervision Detections
            detections_sv = sv.Detections.from_ultralytics(detections)
            
            #Conver goalkeeper to player
            for object_id, class_id in enumerate(detections_sv.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detections_sv.class_id[object_id] = cls_names_inv['player']


            
            #Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detections_sv)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detections_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][track_id] = {"bbox": bbox}
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks


            # Convert to custom format
            # detections_sv =[
            #     (cls_names_inv[class_id], confidence, bbox) for xyxy, confidence, class_id, tracker_id in detections_sv
            # ]
            # yield frame_num, detections_sv


    # def draw_ellipse(self, frame, bbox, color, track_id=None):
    # # Calculate center and width
    #     x_center, y_center = get_center_of_bbox(bbox)
    #     width = get_bbox_width(bbox)
    #     height = int(0.35 * width)  # Adjust height as needed

    #     # Draw the ellipse
    #     cv2.ellipse(
    #         frame,
    #         center=(100, 100),
    #         axes=(50, 20),
    #         angle=0,
    #         startAngle=0,
    #         endAngle=360,
    #         color=(255, 0, 0),
    #         thickness=2
    #     )

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
    
    def draw_triangle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ], np.int32)

        cv2.drawContours(frame, [triangle_points], 0, color,  cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    #     # Optional: draw a dot at the center for visual confirmation
    #     cv2.circle(frame, (x_center, y_center), 5, (0, 255, 255), -1)  # Yellow dot
    #     return frame

        
    def draw_annotations(self, video_frames, tracks):
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

            output_video_frames.append(frame)

        return output_video_frames

