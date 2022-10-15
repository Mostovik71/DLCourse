import easyocr
reader = easyocr.Reader(['en'], recog_network='custom_example')
result = reader.readtext('cropped.jpg')
print(result)
