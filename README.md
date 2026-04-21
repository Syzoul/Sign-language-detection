Introduction to our AI Model

Authors

Le Thanh Sang

Bui Dinh Khoi

Nguyen Huu Cat Tuong


Welcome to our AI model, designed to recognize sign language from images. This project uses Mediapipe technology to extract the coordinates of hand landmarks, allowing the model to identify alphabet characters.

The model uses the MediaPipe Hands library developed by Google to detect and extract hand features from input images. Instead of using raw image data for training, the proposed method uses the spatial coordinates of hand landmarks as input features. This representation reduces the data dimensionality, which helps shorten the training time and simplify the model. Despite using a simpler input representation, the model still achieves high recognition performance.
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/8bc251df-1659-4b32-a131-110b6d320477" />

Key Features:

Mediapipe Technology: Accurately detects and extracts hand landmarks, unaffected by lighting or background variations.

MLP Neural Network Model: Uses MLP to classify sign language gestures into alphabet characters.

Advantages of Landmarks: Landmarks provide accuracy, fast and simple training, and robustness against lighting, background, and hand shape variations.

**Note:
Limitations: The model is trained on static images and does not recognize the letters J and Z.**

We hope this AI model will greatly benefit your work!


The following is the hand gesture alphabet used in the model.
<img width="760" height="430" alt="image" src="https://github.com/user-attachments/assets/3c1e36a4-35e7-4df3-a7d2-137850560063" />




