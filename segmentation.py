import torch
from PIL import Image
import io
from PIL import UnidentifiedImageError


def get_yolov7():
    # local best.pt
    model = torch.hub.load('./yolov7', 'custom', path_or_model='./model/best.pt', source='local', force_reload=True)  # local repo
    model.conf = 0.5
    return model


def get_image_from_bytes(binary_image, max_size=1024):
    try:
        input_image = Image.open(io.BytesIO(binary_image))#.convert("RGB")
    except UnidentifiedImageError:
        return 0
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(720),
            int(720),
        )
    )
    return resized_image
