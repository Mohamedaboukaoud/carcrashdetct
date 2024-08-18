import cv2
import numpy as np

# Paths to YOLO files
coco_names = "/Users/mohamedaboukaoud/Downloads/project 4/coco.names"
yolo_weights = "/Users/mohamedaboukaoud/Downloads/yolov3.weights"
yolo_cfg = "/Users/mohamedaboukaoud/Downloads/yolov3.cfg"

def load_yolo():
    # Load YOLO
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    with open(coco_names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get the names of all the layers
    layer_names = net.getLayerNames()
    print(f"Layer names: {layer_names}")  # Debug: Print all layer names

    # Get the indices of the output layers
    output_layer_indices = net.getUnconnectedOutLayers()
    print(f"Unconnected Out Layers Indices: {output_layer_indices}")  # Debug: Print output layer indices

    # Adjust indices to zero-based indexing
    output_layer_indices = [i - 1 for i in output_layer_indices]
    output_layers = [layer_names[i] for i in output_layer_indices]
    print(f"Output Layers: {output_layers}")  # Debug: Print output layer names
    
    return net, classes, output_layers
def process_frame(frame, net, output_layers, classes):
    height, width = frame.shape[:2]
    
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get the outputs from the network
    outputs = net.forward(output_layers)

    # Debug: Print shapes of outputs
    print("Output shapes:")
    for output in outputs:
        print(output.shape)
    
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            for obj in detection:
                # Ensure obj is a 1D array of length 85
                if len(obj) == 85:  # Check the length of each detection
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    
                    # Extract scores and class ID
                    scores = obj[5:]
                    if len(scores) > 0:
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        # Only consider detections with high confidence
                        if confidence > 0.5:
                            # Calculate bounding box coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
    
    # Print boxes, confidences, and class_ids for debugging
    print("Boxes:", boxes)
    print("Confidences:", confidences)
    print("Class IDs:", class_ids)
    
    # Apply Non-Maximum Suppression to remove redundant overlapping boxes
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    # Load YOLO
    net, classes, output_layers = load_yolo()
    
    # Open video file or capture from camera
    cap = cv2.VideoCapture("/Users/mohamedaboukaoud/Downloads/project/caraccident.mp4")  # Replace 'video.mp4' with your video file path or use 0 for webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process each frame
        frame = process_frame(frame, net, output_layers, classes)
        
        # Display the frame with detections
        cv2.imshow("Frame", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
