import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
from plate_worker.tools import unzip


class OpencvImageLoader():
    def load(self, img_path):
        img = cv2.imread(img_path)
        img = img[..., ::-1]
        return img


inputs = ["44.jpg"]
image_loader = OpencvImageLoader()
inputs = [image_loader.load(item) for item in inputs]

print(inputs)
from plate_worker.yolo_v5_detector import Detector
detector = Detector()
detector.load("latest")
model_outputs = detector.predict(inputs)
images_bboxs, images = [model_outputs, inputs]
print(images_bboxs)


from plate_worker.key_points_detector import NpPointsCraft
detector = NpPointsCraft()
detector.load("latest", "latest")

_inputs = unzip([images, images_bboxs])
_inputs = detector.preprocess(_inputs, **{})
_inputs = detector.forward_batch(_inputs, **{})
_inputs = unzip(detector.postprocess(_inputs, **{}))
images_points, images_mline_boxes = unzip(_inputs)

from plate_worker.image_utils import crop_number_plate_zones_from_images

zones, image_ids = crop_number_plate_zones_from_images(images, images_points)


import cv2
for i, im in enumerate(zones):
    cv2.imshow("d", im)
    #cv2.imwrite(f"to_ocr_{str(i)}.jpg", im)
    k = cv2.waitKey(0)
    if k == ord("q"):
        continue