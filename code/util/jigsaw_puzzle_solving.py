import cv2
import numpy as np
import random

class JigsawPuzzleSolving:
    def __init__(self, image_path, rows=3, cols=3):
        self.image = cv2.imread(image_path)
        self.rows = rows
        self.cols = cols
        if self.image is None:
            raise ValueError
        self.tiles = self.split_image_into_tiles()

    def split_image_into_tiles(self):
        """ 이미지를 지정된 수의 행과 열로 나눕니다. """
        tile_height, tile_width = self.image.shape[0] // self.rows, self.image.shape[1] // self.cols
        return [self.image[r*tile_height:(r+1)*tile_height, c*tile_width:(c+1)*tile_width]
                           for r in range(self.rows) for c in range(self.cols)]


    def shuffle_and_reconstruct(self):
        """ 타일을 섞고 이미지 재구성 후 결과 반환. """
        random.shuffle(self.tiles)
        new_image = np.vstack([np.hstack(self.tiles[i*self.cols:(i+1)*self.cols]) for i in range(self.rows)])
        return new_image
