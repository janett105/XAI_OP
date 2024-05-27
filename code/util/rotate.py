import os
from PIL import Image

# 입력 폴더와 출력 폴더 경로
input_folder = 'test_to/0'
output_folder = 'test_to/0_rotated'

# 입력 폴더와 출력 폴더 경로 설정 (절대 경로 사용)
input_folder = r'C:\Users\xsamk\Desktop\4학년1학기\caps\data_rotated\DB_X-ray\validation_to/1'
output_folder = r'C:\Users\xsamk\Desktop\4학년1학기\caps\data_rotated\DB_X-ray\validation_to/1_rotated'

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 지원하는 이미지 파일 확장자
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# 입력 폴더의 모든 파일에 대해 작업 수행
for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):
        # 이미지 파일 경로
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # 이미지 열기
        image = Image.open(input_path)
        
        # 이미지 좌우 반전
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 반전된 이미지 저장
        flipped_image.save(output_path)

        print(f"{filename}을(를) 반전시켜 {output_path}에 저장했습니다.")

print("모든 이미지 파일의 반전 작업이 완료되었습니다.")
