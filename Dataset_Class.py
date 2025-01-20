import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms # 데이터 전처리를 위한 모듈
from PIL import Image # 이미지 파일을 불러오기 위한 모듈

class PyTorch_Custom_Dataset_Class(Dataset):
    def __init__(self): # 데이터 세트 클래스를 상속 받은 후 초기화, 데이터 세트 클래스에서 사용하고자 하는 변수 초기화
        super().__init__()
        pass
    def __getitem__(self, idx): # 이미지와 정답 반환. 이미지는 PIL로 읽어오고 정답은 클래스 이름 리스트에서 해당 클래스에 해당하는 인덱스 반환환
        pass
    def __len__(self): # 데이터 세트 크기 반환
        pass
        
        
class PyTorch_Classification_Dataset_Class(Dataset):
    def __init__(self 
                , dataset_dir = "/content/Recycle_Classification_Dataset"
                , transform = None): # 데이터세트 경로, 데이터세트 전처리 방법 입력
        super().__init__()
        # 데이터 세트가 지정한 경로에 있는지 확인
        if not os.path.isdir(dataset_dir):
            # 데이터 세트가 없다면 깃허브에서 데이터 세트를 다운로드
            os.system("git clone https://github.com/JinFree/Recycle_Classification_Dataset.git")
            os.system("rm -rf ./Recycle_Classification_Dataset/.git")
        
        self.image_abs_path = dataset_dir
        # 이미지를 신경망에 입력하기 전 전처리 방법을 정의
        # 기본 값은 None
        self.transform = transform
        # 전처리 방법이 없다면 사전에 입력한 전처리 방법을 선언
        if self.transform is None:
            # 사전에 입력한 전처리 방법
            self.transform = transforms.Compose([
                    transforms.Resize(256) # 이미지를 256x256으로 변환
                    , transforms.RandomCrop(224) # 이미지를 224x224로 랜덤하게 자름
                    , transforms.ToTensor() # 0 ~ 255 값을 가진 데이터를 0 ~ 1 사이의 값을 가진 텐서로 변환
                    , transforms.Normalize(mean=[0.485, 0.456, 0.406], # 정규화
                            std=[0.229, 0.224, 0.225]) 
                    ])
        # 입력받은 경로 바로 아래에 있는 폴더의 이름이 분류할 클래스의 이름이 됨
        self.label_list = os.listdir(self.image_abs_path) # 클래스 이름 리스트 생성
        self.label_list.sort()
        self.x_list = [] # 이미지 데이터의 경로를 보관할 리스트 생성
        self.y_list = [] # 각 이미지의 클래스 정보를 보관할 리스트 생성
        # self.image_abs_path 경로 내 모든 폴더를 순차적으로 확인
        for label_index, label_str in enumerate(self.label_list):
            # 이미지만 있다고 가정하고 폴더 내 모든 파일의 경로를 리스트로 제작
            img_path = os.path.join(self.image_abs_path, label_str)
            img_list = os.listdir(img_path)
            for img in img_list:
                self.x_list.append(os.path.join(img_path, img))
                self.y_list.append(label_index)

    def __len__(self): # 데이터 길이 반환
        return len(self.x_list)

    def __getitem__(self, idx): # 데이터의 인덱스를 입력받고 클래스 정보를 반환
        # PIL 모듈의 Image 함수를 활용해 이미지를 열기
        image = Image.open(self.x_list[idx])
        # 이미지가 흑백이 아니면 컬러로 변환
        if image.mode is not "RGB":
            image = image.convert('RGB')
        # 데이터 세트를 신경망에 입력하기 전에 전처리를 수행
        if self.transform is not None:
            image = self.transform(image)
        return image, self.y_list[idx]

    def __save_label_map__(self, dst_text_path = "label_map.txt"): # 이미지의 클래스 저장
        label_list = self.label_list
        f = open(dst_text_path, 'w')
        for i in range(len(label_list)):
            f.write(label_list[i]+'\n')
        f.close()

    def __num_classes__(self):
        return len(self.label_list)
