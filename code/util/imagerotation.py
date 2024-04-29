import torchvision.transforms.functional as TF
import random
from PIL import Image

    
def rotate_img(img, angle):
    """이미지를 주어진 각도로 회전합니다."""
    if angle == 0:
        return img
    elif angle == 90:
        return img.transpose(Image.ROTATE_90)
    elif angle == 180:
        return img.transpose(Image.ROTATE_180)
    elif angle == 270:
        return img.transpose(Image.ROTATE_270)
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class ImageRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, x):
        angle = random.choice(self.degrees)
        return TF.rotate(x, angle)

