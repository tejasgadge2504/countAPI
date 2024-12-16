from flask import Flask, request, jsonify
import cv2
import numpy as np
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import base64
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Allow CORS for all domains

# Load the gender detection model
gender_model = load_model('models/gender_detection.h5')
gender_classes = ['man', 'woman']

# Load YOLO pre-trained model
weights_path = "models/yolov3.weights"
config_path = "models/yolov3.cfg"
names_path = "models/coco.names"

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load COCO class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the frame from the request (Base64 encoded image)
        data = request.get_json()
        image_data = data.get("frame")
        
        if not image_data:
            return jsonify({"error": "No frame provided"}), 400

        # Decode the base64 image to raw image format
        decoded_data = base64.b64decode(image_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        height, width, channels = frame.shape

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Arrays for detected objects
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        man_count = 0
        woman_count = 0

        for i in indexes:
            x, y, w, h = boxes[i]
            person_img = frame[y:y+h, x:x+w]

            if person_img.size == 0:
                continue

            faces, _ = cv.detect_face(person_img)
            for face in faces:
                startX, startY, endX, endY = face
                face_crop = person_img[startY:endY, startX:endX]

                if face_crop.shape[0] >= 10 and face_crop.shape[1] >= 10:
                    face_crop = cv2.resize(face_crop, (96, 96))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)
                    conf = gender_model.predict(face_crop)[0]
                    gender_label = gender_classes[np.argmax(conf)]
                    if gender_label == 'man':
                        man_count += 1
                    else:
                        woman_count += 1

        response = {
            "man_count": man_count,
            "woman_count": woman_count
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
