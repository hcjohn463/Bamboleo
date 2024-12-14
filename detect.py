import argparse
from ultralytics import YOLO
import cv2
import math
import os

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    _, ext = os.path.splitext(filename)
    return ext.lower() in video_extensions

def process_image(frame, block_model, plate_model):
    # 對圖片進行偵測
    block_results = block_model.predict(
        source=frame,
        show=False,
        save=True,
        conf=0.5
    )
    
    plate_results = plate_model.predict(
        source=frame,
        show=False,
        save=True,
        conf=0.5
    )
    
    # 解析偵測結果
    current_blocks = []
    current_plates = []
    
    for result in block_results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            block = {
                "type": block_model.names[int(class_id)],
                "position": {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                },
                "confidence": round(confidence, 2),
            }
            current_blocks.append(block)

    for result in plate_results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            plate = {
                "type": plate_model.names[int(class_id)],
                "position": {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                },
                "length": y2 - y1,
                "width": x2 - x1,
                "confidence": round(confidence, 2),
            }
            current_plates.append(plate)
            
    return current_blocks, current_plates

def draw_detections(frame, blocks, plates):
    if not plates:
        return frame
        
    plate = plates[0]
    circle_center_x = plate['position']['x']
    circle_center_y = plate['position']['y']
    circle_image_diameter = max(plate['length'], plate['width'])
    circle_real_diameter = 36  # cm
    
    for block in blocks:
        object_center_x = block['position']['x']
        object_center_y = block['position']['y']
        
        # 計算距離
        image_distance = math.sqrt(
            (circle_center_x - object_center_x) ** 2 +
            (circle_center_y - object_center_y) ** 2
        )
        
        scale_ratio = circle_real_diameter / circle_image_diameter
        real_distance = image_distance * scale_ratio
        
        # 繪製圓形和線條
        cv2.circle(frame, (int(circle_center_x), int(circle_center_y)), 
                  5, (0, 0, 255), -1)
        cv2.circle(frame, (int(object_center_x), int(object_center_y)), 
                  5, (0, 0, 255), -1)
        cv2.line(frame, (int(circle_center_x), int(circle_center_y)),
                (int(object_center_x), int(object_center_y)), (0, 0, 255), 2)
        
        # 添加距離文字
        mid_x = (circle_center_x + object_center_x) / 2
        mid_y = (circle_center_y + object_center_y) / 2
        cv2.putText(frame, f"{real_distance:.2f} cm", 
                   (int(mid_x), int(mid_y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def detect(opt):
    # Load pretrained YOLO models
    block_model = YOLO("block.pt")
    plate_model = YOLO("plate.pt")

    if opt.source == "camera":
        # Process camera type
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_blocks, current_plates = process_image(frame, block_model, plate_model)
            frame = draw_detections(frame, current_blocks, current_plates)
            
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        
    elif is_video_file(opt.source):
        # Process video type
        cap = cv2.VideoCapture(opt.source)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = 'labeled_' + opt.source
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}")
            
            current_blocks, current_plates = process_image(frame, block_model, plate_model)
            frame = draw_detections(frame, current_blocks, current_plates)
            
            cv2.imshow("Detections", frame)
            out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        print(f"Video processing is complete and has been saved to {output_path}")
        
    else:
        # Process image type
        frame = cv2.imread(opt.source)
        if frame is None:
            print(f"Can't read image: {opt.source}")
            return
            
        current_blocks, current_plates = process_image(frame, block_model, plate_model)
        frame = draw_detections(frame, current_blocks, current_plates)
        
        # end
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        
        # save result
        output_path = 'labeled_' + opt.source
        cv2.imwrite(output_path, frame)
        print(f"Image processing is complete and has been saved to {output_path}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bamboleo Detection Script")
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Path to the input image, video file, or 'camera' for live detection"
    )

    opt = parser.parse_args()
    detect(opt)