import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

from .Model_Class_From_the_Scratch import MODEL_From_Scratch
from .Model_Class_Transfer_Learning_MobileNet import MobileNet
from .Dataset_Class import PyTorch_Classification_Dataset_Class as Dataset

class PyTorch_Classification_Training_Class():
    def __init__(self 
                , dataset_dir = "/content/Recycle_Classification_Dataset"
                , batch_size = 16
                , train_ratio = 0.75
                ): # 학습을 위한 초기화
        # 데이터세트가 경로에 없는 경우 깃허브에서 데이터세트를 다운로드해 데이터세트를 준비비
        if not os.path.isdir(dataset_dir):
            os.system("git clone https://github.com/JinFree/Recycle_Classification_Dataset.git")
            os.system("rm -rf ./Recycle_Classification_Dataset/.git")
            dataset_dir = os.path.join(os.getcwd(), 'Recycle_Classification_Dataset')  
        # CUDA를 사용할 수 있는 경우 self.USE_CUDA를 True로 설정
        self.USE_CUDA = torch.cuda.is_available()
        # CUDA를 사용할 수 있는 경우 self.DEVICE를 "cuda"로 설정
        self.DEVICE = torch.device("cuda" if self.USE_CUDA else "cpu")
        # Dataset Class에서와 같은 전처리 선언
        self.transform = transforms.Compose([
                transforms.Resize(256)
                , transforms.RandomCrop(224)
                , transforms.ToTensor()
                , transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]) 
                ])
        # Dataset Class를 생성 및 초기화
        dataset = Dataset(dataset_dir = dataset_dir, transform = self.transform)
        # 라벨 파일 저장
        dataset.__save_label_map__()
        self.num_classes = dataset.__num_classes__()
        # 데이터 세트를 훈련용 데이터세트와 검증용 데이터 세트로 분류
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        # 데이터 로더를 생성, shuffle=True로 설정하여 데이터를 섞음
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset
            , batch_size=batch_size
            , shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset
            , batch_size=batch_size
            , shuffle=False
        )
        # 신경망 모델과 모델 이름 초기화
        self.model = None
        self.model_str = None
        
    def prepare_network(self
            , is_scratch = True): # 직접 구현한 신경망을 학습할 것인지, 전이학습을 위한 신경망을 학습한 것인지에 따른 신경망 초기화
        if is_scratch:
            # is_scratch가 True인 경우, 직접 구현한 신경망을 이용
            self.model = MODEL_From_Scratch(self.num_classes)
            self.model_str = "PyTorch_Training_From_Scratch"
        else: # is_scratch가 False인 경우, 전이학습으로 훈련한 신경망을 이용
            self.model = MobileNet(self.num_classes)
            self.model_str = "PyTorch_Transfer_Learning_MobileNet"
        self.model.to(self.DEVICE)
        self.model_str += ".pt" 
    
    def training_network(self
            , learning_rate = 0.0001
            , epochs = 10
            , step_size = 3
            , gamma = 0.3): # 실제 훈련 수행
        if self.model is None:
            self.prepare_network(False)
        # 최적화 함수와 스케줄러 초기화
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # 스케줄러는 step_size 에포크마다 학습률에 gamma를 곱함
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        acc = 0.0
        for epoch in range(1, epochs + 1):
            self.model.train()
            for data, target in tqdm(self.train_loader):
                # data가 이미지, target이 정답
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                # 학습을 위한 미분값을 0으로 초기화
                optimizer.zero_grad()
                # 모델 계산을 수행
                output = self.model(data)
                # 손실함수 계산 수행
                loss = F.cross_entropy(output, target)
                # 역전파 계산을 수행
                loss.backward()
                # 최적화 수행
                optimizer.step()
            # 학습률 스케줄러 업데이트
            # 현 에포크가 끝난 후 수행
            scheduler.step()
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad(): # 검증 과정에서는 기울기를 계산하지 않음
                for data, target in tqdm(self.test_loader):
                    data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                    output = self.model(data)
                    # 교차 엔트로피를 통해 손실값을 계산해 test_loss에 누적
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    # outpout에서 가장 확률이 높은 것의 인덱스를 받아와 pred에 저장
                    pred = output.max(1, keepdim=True)[1]
                    # 정답과 예측값이 동일하면 correct에 값을 추가
                    correct += pred.eq(target.view_as(pred)).sum().item()
            # 평균 손실값과 정확도 출력력
            test_loss /= len(self.test_loader.dataset)
            test_accuracy = 100. * correct / len(self.test_loader.dataset)
            print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
            # 검증용 데이터세트의 정확도가 좋아지거나 마지막 에포크인 경우 이 모델을 저장
            if acc < test_accuracy or epoch == epochs:
                acc = test_accuracy
                torch.save(self.model.state_dict(), self.model_str)
                print("model saved!")
        
if __name__ == "__main__":
    training_class = PyTorch_Classification_Training_Class()
    training_class.prepare_network(True)
    training_class.training_network()
