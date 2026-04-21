Introduction to Our AI Model
Authors
Le Thanh Sang
Bui Dinh Khoi
Nguyen Huu Cat Tuong
Overview

This project presents an AI model for recognizing sign language alphabets from images. The system uses the MediaPipe Hands library developed by Google to extract hand landmark coordinates, which are then used as input features for classification.

Instead of using raw images, the model relies on hand landmark coordinates. This approach reduces data dimensionality, speeds up training, and simplifies the model while maintaining high recognition accuracy.

Methodology
Hand detection and feature extraction are performed using MediaPipe Hands
A Multi-Layer Perceptron (MLP) is used for classification
Input features: 21 hand landmarks (x, y, z coordinates)
Key Features
Efficient Training: Uses low-dimensional landmark data instead of images
High Performance: Maintains strong accuracy despite simpler input
Robustness: Less sensitive to lighting, background, and hand variations
Limitations
Only supports static gestures
Does not recognize dynamic letters such as J and Z


Hand Gesture Alphabet

<img width="760" height="430" alt="image" src="https://github.com/user-attachments/assets/d40c28c9-9088-47a2-b30b-7061579d3bde" />




Conclusion
This model demonstrates that using hand landmarks is an efficient and effective approach for sign language recognition. It reduces computational cost while still achieving reliable performance.

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/3530952e-51d9-406c-a410-c9ff30d8aef2" />

