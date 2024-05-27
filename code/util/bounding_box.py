from PIL import Image, ImageDraw
import os
from torchvision.transforms import Resize
import numpy as np

def createBBox(base_path):
    input_size = 224

    total_bbox=np.zeros((input_size, input_size))

    Bbox_path = base_path+'Bbox.npy'
    for path1 in ['train/', 'test/','validation/']:
        for path2 in ['0/', '1/']:
            images_path = base_path+path1+path2
            for file_name in os.listdir(images_path):
                img = Image.open(images_path+file_name)

                # 이미지 resize (224, 224)
                img = img.resize((input_size, input_size))
                
                # bouding box 안 1, 밖 0으로 설정
                img_np = np.array(img)
                img_np = np.where(img_np > 0, 1, 0)

                # 전체 bounding box 합침
                try:
                    total_bbox += img_np
                except:
                    img_np = img_np[:, :,0]
                    total_bbox += img_np
  
    # boudning box밖은 0이 아닌 최솟값으로 진행
    min_value = min(total_bbox[total_bbox != 0])
    np.where(total_bbox==0, min_value, total_bbox) 

    # 확률 분포로 변환
    total_bbox = total_bbox / np.sum(total_bbox)
    print(total_bbox)

    # 저장    
    np.save(Bbox_path, total_bbox)
    # totalBbox_img = Image.fromarray(total_bbox)
    # totalBbox_img= totalBbox_img.convert('RGB')
    # totalBbox_img.save(base_path+'Bbox.png')
    
    return total_bbox

def createBBoxline(base_path):
    Bbox = np.load(base_path+'Bbox.npy')  
    Bbox_line = np.where(Bbox> 0, 1, 0)
    
    # BoxPixels = np.where(Bbox_line==1)
    # left = min(BoxPixels[0])
    # right = max(BoxPixels[0])
    # bottom = min(BoxPixels[1])
    # top = max(BoxPixels[1])

    # Bbox_line[bottom:top, left:right] == 1

    Bbox_line_img = Image.fromarray(np.where(Bbox_line>0, 255, 0)).convert('RGB')
    # np.save(base_path+'BboxLine.npy', Bbox_line)
    Bbox_line_img.save(base_path+'BboxLine!.jpg')

    return Bbox_line

base_path = 'C:/Users/janet/MyProjects/XAI_OP/data/DB_BBox/'
total_bbox = createBBox(base_path)
# Bbox_line = createBBoxline(base_path)