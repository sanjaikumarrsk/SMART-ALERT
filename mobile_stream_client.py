import cv2
import requests

FLASK_URL = "http://127.0.0.1:8000/predict"

# PUT YOUR MOBILE IP CAMERA ADDRESS HERE
MOBILE_CAM_URL = "http://172.20.10.8:8080/video"

cap = cv2.VideoCapture(MOBILE_CAM_URL)

import cv2
import requests

FLASK_URL = "http://127.0.0.1:8000/predict"

# PUT YOUR MOBILE IP CAMERA ADDRESS HERE
MOBILE_CAM_URL = "http://172.20.10.8:8080/video"

cap = cv2.VideoCapture(MOBILE_CAM_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Reconnecting...")
        cap = cv2.VideoCapture(MOBILE_CAM_URL)
        continue

    # Encode frame as JPEG
    _, jpeg = cv2.imencode(".jpg", frame)

    # SEND FRAME TO FLASK
    res = requests.post(
        FLASK_URL,
        files={"image": ("frame.jpg", jpeg.tobytes(), "image/jpeg")}
    )

    try:
        print(res.json())
    except Exception:
        print('No JSON response, status', getattr(res, 'status_code', None))

    # Show image locally
    cv2.imshow("Mobile Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break