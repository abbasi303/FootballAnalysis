import cv2

# Function to read frames from the video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Append individual frame, not the list itself
    cap.release()  # Release the video capture to free resources
    return frames

# Function to save video from frames
def save_video(output_video_frames, output_video_path):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = output_video_frames[0].shape[:2]  # Extract height and width from the first frame
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in output_video_frames:
        out.write(frame)  # Write each frame to the output video

    out.release()  # Release the video writer to save the file properly

