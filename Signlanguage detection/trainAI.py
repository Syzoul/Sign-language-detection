import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from IPython.display import clear_output

# Khởi tạo Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Hàm để lấy tọa độ các điểm landmarks của bàn tay và lấy cổ tay làm gốc tọa độ
def get_hand_landmarks(image, num_landmarks=21):
    landmarks = []
    default_value = (-1, -1)  # Giá trị mặc định cho điểm bị thiếu
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = image.shape
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            hand_landmark_coords = [default_value] * num_landmarks  # Khởi tạo với giá trị mặc định
            for id, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * w) - wrist_x, int(lm.y * h) - wrist_y
                hand_landmark_coords[id] = (x, y)
            landmarks.append(hand_landmark_coords)
    return landmarks

# Đường dẫn tới Mnist folder
# Mnist folder's path
mnist_folder = Path('D:\My code\Python\Mnist')

# Get the trainingSet and testSet folders
training_set_folder = mnist_folder / 'trainingSet' / 'trainingSet'

# Hàm để liệt kê tất cả các đường dẫn tệp ảnh trong từng thư mục con
def list_image_file_paths(folder):
    image_file_paths = {}
    total_images = 0
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            files = [f for f in subfolder.iterdir() if f.is_file()]
            image_file_paths[subfolder.name] = files
            total_images += len(files)  # Đếm số lượng tệp
    return image_file_paths, total_images

# Liệt kê các đường dẫn tệp ảnh trong trainingSet
training_set_image_paths, training_set_total_images = list_image_file_paths(training_set_folder)
print("Number of images:", training_set_total_images)

# Tạo training data và target
train_data = []
train_target = []

# Load tất cả các ảnh có thể tốn nhiều bộ nhớ, bạn có thể chia thành các lô nhỏ
for subfolder, files in training_set_image_paths.items():
    clear_output(wait=True)  # Xóa dòng đã in, thay bằng dòng mới
    print(f"Processing folder: {subfolder}")
    print(f"Number of images: {len(files)}")
    # Chuẩn bị dữ liệu huấn luyện
    for i, file_path in enumerate(files):
        # Đọc ảnh và lấy các hand landmarks
        image_path = str(file_path)
        img = cv2.imread(image_path)
        if img is not None:
            landmarks = get_hand_landmarks(img)
            if landmarks and len(landmarks[0]) == 21:
                train_data.append([coord for point in landmarks[0] for coord in point])  # Làm phẳng và thêm vào train_data
                train_target.append(int(subfolder))
            print(f"Processing: {i}/{len(files)}\r", end="")
        else:
            print(f"Không thể đọc ảnh {image_path}.")

# Chuyển đổi thành numpy arrays
train_data = np.array(train_data, dtype=np.float32)
train_target = np.array(train_target, dtype=np.uint8)

# Chia và xáo trộn dữ liệu huấn luyện
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.3, shuffle=True)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Xây dựng mô hình mạng Neural
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(400,400,400,400), activation='relu', solver='adam', batch_size=100, max_iter=500)
mlp.fit(X_train, y_train)

# Đánh giá mô hình huấn luyện
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
predictions = mlp.predict(X_test)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# save model for future job
import pickle
# # save to file
filename = 'MLP_model.sav'
pickle.dump(mlp, open(filename, 'wb'))
# # # To load model in another program
# filename = 'MLP_model.sav'
# clf = pickle.load(open(filename, 'rb'))
# clf.predict(test_data)pip install ipython
