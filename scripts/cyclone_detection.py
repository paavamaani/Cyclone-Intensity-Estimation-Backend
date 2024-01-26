from PIL import Image
import torch
from io import BytesIO
import cv2

def detectCyclone(img_bytes):
    img = Image.open(BytesIO(img_bytes))
    model = torch.hub.load('./yolov5', 'custom', path='models/best.pt', force_reload=True, source='local')

    results = model(img, size=640)

    detections = results.xyxy[0].cpu().numpy()

    if len(detections) == 0:
        return [None, None]

    image = results.crop(save=False)[0]['im']
    # cv2.imwrite('./static/roi_images/image.jpg', image)

    results = list(results.xyxy[0][:, :-1].numpy()[0])
    
    confidence = results[-1]

    results = [int(x) for x in results[:4]]
    results.append(confidence)

    return [image, results]