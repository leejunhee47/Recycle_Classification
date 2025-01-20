#!python3
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from Model_Class_From_the_Scratch import MODEL_From_Scratch
from Model_Class_Transfer_Learning_MobileNet import MobileNet
import argparse # 추론을 어떤 신경망에서 할 것인지, 어떤 동영상 스트림을 추론할 것인지를 입력해 실행행
import cv2


class Inference_Class():
    # 추론을 위한 초기화 메소드
    def __init__(self): 
        USE_CUDA = torch.cuda.is_available() # CUDA를 사용할 수 있는지 확인
        self.DEVICE = torch.device("cuda" if USE_CUDA else "cpu") 
        self.model = None # 추론할 신경망 선언
        self.label_map = None # 클래스 이름 리스트 변수 선언
        self.transform_info = transforms.Compose([
                transforms.Resize(size=(224, 224)), # 224 * 224로 리사이즈
                transforms.ToTensor()
                ])


    def load_model(self, is_train_from_scratch, label_map_file = "label_map.txt"):
        self.label_map = np.loadtxt(label_map_file, str, delimiter='\t') # 클래스 이름 리스트 불러오기
        num_classes = len(self.label_map)
        model_str = None
        if is_train_from_scratch: # --is_scratch를 추가했다면 직접 구현해 훈련했던 신경망과 가중치를 불러온다
            self.model = MODEL_From_Scratch(num_classes).to(self.DEVICE)
            model_str = "PyTorch_Training_From_Scratch"
        else: # 그렇지 않았다면 MobileNetv2에서 전이학습의 신경망과 가중치를 불러온다
            self.model = MobileNet(num_classes).to(self.DEVICE)
            model_str = "PyTorch_Transfer_Learning_MobileNet"
        model_str += ".pt"
        self.model.load_state_dict(torch.load(model_str, map_location=self.DEVICE))
        self.model.eval()
        
        
    def inference_video(self, video_source="test_video.mp4"): # 영상 추론 함수
        cap = cv2.VideoCapture(video_source) 
        # OpenCV가 비디오를 정상적으로 열 수 있는지 확인
        if cap.isOpened():
            print("Video Opened")
        else: # 열지 못하면 실행 중지
            print("Video Not Opened")
            print("Program Abort")
            exit()
        # Output이라는 이름을 가진 창을 생성하고 추론 결과를 표시한 동영상을 보여준다.
        cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
        with torch.no_grad(): # 가중치 학습 비활성화
            while cap.isOpened():
                ret, frame = cap.read() # 프레임을 받아옴. ret : 프레임을 정상적으로 받아왔는지 True or False. frame : 실제 동영상 프레임
                if ret:
                    output = self.inference_frame(frame)
                    cv2.imshow("Output", output) # 이후 후처리까지 끝난 결과 프레임을 output으로 받아와서 Output에 보여준다.
                else:
                    break
                # 동영상을 30FPS로 보여주다가 'q'를 입력하면 반복문 종료
                if cv2.waitKey(33) & 0xFF == ord('q'): # 33ms 동안 이미지를 보여줌. 30프레임을 보기 위해서는 이미지 한 장당 33ms만 보여주면 되므로 이렇게 작성
                    break
            cap.release()
            cv2.destroyAllWindows() # 반복문이 종료되면 동영상 스트림을 닫고 열려 있는 모든 창을 닫은 후 함수 실행을 종료.
        return
    

    def inference_frame(self, opencv_frame): # 프레임을 추론
        # OpenCV는 영상 데이터를 BGR 순서로 관리. PIL.Image는 RGB 순서로 관리. 
        # OpenCV를 PIL로 변환하기 위해서 BGR에서 RGB로 바꿔야함
        opencv_rgb = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB) 
        image = Image.fromarray(opencv_rgb)
        image_tensor = self.transform_info(image) # 초기화한 전처리 수행
        image_tensor = image_tensor.unsqueeze(0) # 4차원 텐서로 변환
        image_tensor = image_tensor.to(self.DEVICE) # self.DEVICE로 데이터 이동
        inference_result = self.model(image_tensor)
        inference_result = inference_result.squeeze()
        inference_result = inference_result.cpu().numpy()
        result_frame = np.copy(opencv_frame)
        label_text = self.label_map[np.argmax(inference_result)]
        label_text += " " + str(inference_result[np.argmax(inference_result)])
        result_frame = cv2.putText(result_frame, label_text, (10, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, color=(0,0,255), thickness=3)
        return result_frame
    
    
    def inference_image(self, opencv_image):
        opencv_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(opencv_rgb)
        image_tensor = self.transform_info(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.DEVICE)
        with torch.no_grad():
            inference_result = self.model(image_tensor)
        inference_result = inference_result.squeeze() # 추론 결과를 squeeze로 차원을 줄임
        inference_result = inference_result.cpu().numpy() # numpy 형태로 변환 ["클래스 0일 확률", "1일 확률", "2일 확률", "3일 확률"]
        result_frame = np.copy(opencv_image)
        label_text = self.label_map[np.argmax(inference_result)] # 클래스 이름을 훈련할 때 저장, 추론할 때 불러와서 활용
        class_prob = str(inference_result[np.argmax(inference_result)]) 
        # putText : 사람이 읽을 수 있게 출력 프레임에 해당 정보를 표시
        result_frame = cv2.putText(result_frame, label_text + " " + class_prob, (10, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, color=(0,0,255), thickness=3)
        return result_frame, label_text, class_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--is_scratch", 
            required=False,
            action="store_true", 
            help="inference with model trained from the scratch") # --is_scatch를 입력하면 직접 구현해 학습한 신경망 이용, 입력하지 않으면 전이학습으로 훈련한 신경망 이용
    parser.add_argument("-src", "--source", 
            required=False,
            type=str, 
            default="./test_video.mp4", 
            help="OpenCV Video source") # --source 뒤에 추론할 비디오 파일의 경로를 입력한다. 생략할 경우 기본으로 test_video.mp4를 추론
    args = parser.parse_args()
    is_train_from_scratch = False
    source = args.source
    if args.is_scratch:
        is_train_from_scratch = True
    inferenceClass = Inference_Class()
    inferenceClass.load_model(is_train_from_scratch)
    inferenceClass.inference_video(source)
