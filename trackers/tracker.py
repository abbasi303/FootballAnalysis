from ultralytics import YOLO
import supervision as sv


class Tracker:
    def __init__(self, model_path):
        self.model=YOLO(model_path)

    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0, len(frames),batch_size):
            detections_batch= self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections = detections_batch
        return detections


    def get_object_tracks(self, frames):
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
                if cls_names_inv[class_id] == 'goalkeeper':
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
            
            
        return tracks


            # Convert to custom format
            # detections_sv =[
            #     (cls_names_inv[class_id], confidence, bbox) for xyxy, confidence, class_id, tracker_id in detections_sv
            # ]
            # yield frame_num, detections_sv