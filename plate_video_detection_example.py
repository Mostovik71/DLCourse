import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2 as cv
from plate_worker.tools import unzip
from plate_worker.yolo_v5_detector import Detector
from plate_worker.key_points_detector import NpPointsCraft
from plate_worker.image_utils import crop_number_plate_zones_from_images
import easyocr


def load_path(img_path):
    img = cv.imread(img_path)
    img = img[..., ::-1]
    return img


def load_img(img):
    img = img[..., ::-1]
    return img


def detect_plate(detector, inputs):
    model_outputs = detector.predict(inputs)
    images_bboxs, images = [model_outputs, inputs]
    return images_bboxs, images


def crop_plate(cropper, images, images_bboxs):
    _inputs = unzip([images, images_bboxs])
    _inputs = cropper.preprocess(_inputs, **{})
    _inputs = cropper.forward_batch(_inputs, **{})
    _inputs = unzip(cropper.postprocess(_inputs, **{}))
    images_points, images_mline_boxes = unzip(_inputs)
    zones, image_ids = crop_number_plate_zones_from_images(images, images_points)
    return zones


def recognition_ocr(rec_model, image):
    result = rec_model.readtext(image)
    return result

detector = Detector()
detector.load("latest")
croper = NpPointsCraft()
croper.load("latest", "latest")
reader = easyocr.Reader(["en"], recog_network="custom_example")


cap = cv.VideoCapture("test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ended")
        break
    frame = load_img(frame)
    image_boxes, images = detect_plate(detector, [frame])
    if len(image_boxes[0]) == 0:
        continue
    zones = crop_plate(croper, images, image_boxes)

    cv.imshow('frame_vid', frame)
    for i, zone in enumerate(zones):
        cv.imshow(f'frame_{str(i)}', zone)
        res = recognition_ocr(reader, zone)
        print(res)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
