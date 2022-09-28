import torch
import easyocr
import cv2
from PIL import Image, ImageOps
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='licenseplates.pt', force_reload=True)
image = Image.open('test3.jpg')
result = model(image)
# result.show()
xyxy = result.xyxy

x, y, w, h = int(xyxy[0][0][0]), int(xyxy[0][0][1]), int(xyxy[0][0][2] - xyxy[0][0][0]), int(
    xyxy[0][0][3] - xyxy[0][0][1])
forcrop = (x, y, x + w, y + h)
crop_img = image.crop(forcrop)
crop_img.save('cropped.jpg')
crop_img = ImageOps.grayscale(crop_img)
reader = easyocr.Reader(['ru'])
result = reader.readtext('cropped.jpg')
print(result)
