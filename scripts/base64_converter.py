import base64
from PIL import Image
from io import BytesIO

def getBase64String(image):
    image = Image.fromarray(image)
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    image_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_string