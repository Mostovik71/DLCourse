!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt  # install
!unzip -q /content/drive/MyDrive/PlatesYOLO.zip -d ../data/
!python /content/yolov5/train.py --img 640 --batch 2 --epochs 20 --data /content/data/data.yaml --weights /content/yolov5/yolov5s.pt --cache