- Trước khi sử dụng file .py cần:
    + Tải đầy đủ các thư viện (OpenCV, Mediapipe, NumPy, scikit-learn, Matplotlib, pickle,...)
    + Sửa đổi các đường dẫn tệp trong file .py cho hợp lí.

Giải thích các tệp:
+ Signlanguage detection.zip là tệp nén bao gồm những thành phần bên dưới (có thể tải về và giải nén ra để sử dụng mô hình)
+ Thư mục Mnist chứa các ảnh để huấn luyện mô hình AI.
+ Tệp trainAI.py sử dụng bộ ảnh có trong thư mục Mnist để huấn luyện mô hình, sau đó xuất ra tệp MLP_model.sav để lưu trữ (tệp MLP_model.sav đã được kèm theo, có thể không cần huấn luyện lại)
+ MLP_model.sav là tệp lưu trữ mô hình, chứa thông tin về mô hình đã được huấn luyện.
+ Tệp Detecthand.py là mã để mở camera để nhận diện, nó dùng mô hình MLP_model.sav để nhận diện ngôn ngữ kí hiệu.
+ grid_image.jpg là hình ảnh cho thấy các chữ cái ngôn ngữ kí hiệu và cách thể hiện chúng qua bàn tay, lưu ý: đoạn mã chỉ nhận diện tay phải; mô hình không nhận diện chữ cái J và Z do chỉ được huấn luyện trên ảnh tĩnh.
+ Screenshot 2024-11-01 203404.png cho thấy độ tầng suất đoán đúng của mô hình, qua đó biết được tính hiệu quả của mô hình.

#################################################################################################################

Before using the .py file:
    Ensure all necessary libraries are installed (OpenCV, Mediapipe, NumPy, scikit-learn, Matplotlib, pickle, etc.).
    Modify the file paths in the .py file as appropriate.

File Descriptions:
      Signlanguage detection.zip: A compressed file containing the following components (can be downloaded and extracted for model usage).
      Mnist folder: Contains images used for training the AI model.
      trainAI.py: Uses the images in the Mnist folder to train the model, then exports the MLP_model.sav file for storage (the MLP_model.sav file is included, so retraining may not be necessary).
      MLP_model.sav: A file that stores the trained model, containing information about the trained model.
      Detecthand.py: Code to open the camera for recognition, using the MLP_model.sav model to recognize sign language.
      grid_image.jpg: An image showing the sign language alphabet and how they are represented through hand gestures. Note: the code only recognizes the right hand; the model does not recognize the letters J and Z as it is trained on static images.
      Screenshot 2024-11-01 203404.png: Shows the model's accuracy rate, indicating the model's effectiveness.

