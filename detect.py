import argparse
from ultralytics import YOLO
from datetime import datetime
import cv2
import time

def detect(opt):
    # Load a pretrained YOLO model
    model = YOLO("best.pt")
    
    # Initialize video capture
    if opt.source == "camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(opt.source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

    # Perform prediction using camera or specified source
    results = model.predict(
        source=0 if opt.source == "camera" else opt.source,
        show=True,
        save=True,
        line_width=2,
        save_txt=True,
        conf=0.5
    )

    # Format the results as a list of blocks
    blocks = []
    for idx, result in enumerate(results):
        # Get timestamp
        if opt.source == "camera":
            # For camera, use current time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            # For video, calculate timestamp based on frame number
            current_frame = idx
            seconds = current_frame / fps
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            timestamp = f"{minutes:02d}:{remaining_seconds:05.2f}"

        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            block = {
                "id": idx + 1,
                "type": model.names[int(class_id)],
                "color": "unknown",  # Replace with logic to identify color if available
                "position": {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "z": 0.0  # Assuming 2D detection
                },
                "confidence": round(confidence, 2),
                "timestamp": timestamp  # Add timestamp information
            }
            blocks.append(block)

    # Clean up
    if cap.isOpened():
        cap.release()

    # Print the blocks in JSON-like format
    print("blocks:", blocks)
    return blocks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bamboleo Detection Script")
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Path to the input image, video file, or 'camera' for live detection"
    )

    opt = parser.parse_args()
    result = detect(opt)
    ###### The result will be like
    #    {
    #        "id": 1,  #Bauklötze ID
    #        "type": "cylinder",  # Bauklötze type, e.g., cylinder, cube, triangle
    #        "color": "red",  # Bauklötze color
    #       "position": {  # Coordinates of the Bauklötze in the field of view
    #            "x": 0.5,
    #            "y": 0.4,
    #            "z": 0.0  # If it's 2D, z can be ignored
    #        },
    #        "confidence": 0.98  # Recognition confidence
    #        "timestamp": 00:08.77  # Timestamp information
    #    },
    #    {
    #        "id": 2,
    #        "type": "cube",
    #        "color": "blue",
    #        "position": {"x": 0.3, "y": 0.6, "z": 0.0},
    #        "confidence": 0.95
    #        "timestamp": 00:13.17
    #    },
    #
