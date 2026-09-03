# EcoSort: Smart AI Waste Classifier & Rewards Platform
EcoSort is an intelligent web application designed to bridge the gap between artificial intelligence and environmental sustainability. It leverages computer vision to automatically classify everyday waste items, educate users with tailored recycling tips, and gamify the recycling process through an interactive rewards and eco-impact tracking system.

# Project Overview & Problem Statement
Traditional waste sorting is manual, time-consuming, prone to human error, and sometimes hazardous (e.g., handling batteries or broken glass). **EcoSort** solves this by offering an automated computer vision solution that identifies waste categories instantly. Combined with user authentication, a point-based reward mechanism, and CO₂ savings metrics, it encourages communities to recycle more effectively.

# Dataset & Preprocessing Pipeline (`eda.py` & `train.py`)
The project utilizes a comprehensive garbage classification dataset structured into **7 distinct categories**:
1. **Battery** 
2. **Cardboard** 
3. **Clothes** 
4. **Glass** 
5. **Metal** 
6. **Paper** 
7. **Plastic** 

# Data Processing Steps:
* **Dataset Splitting:** Data is split into **80% for training** and **20% for validation** using a fixed seed (`seed=123`) to ensure consistency and reliable evaluation metrics.
* **Image Resizing & Normalization:** All images are standardized to dimensions of `224x224` pixels and pixel values are normalized to the range `[-1, 1]`.
* **Live Data Augmentation:** Built directly into the Keras sequential pipeline using `RandomFlip`, `RandomRotation (0.2)`, and `RandomZoom (0.2)` to ensure the model generalizes well across diverse real-world angles and lighting conditions.
* **Performance Boosting:** Optimized with TensorFlow's `AUTOTUNE` and `.cache()` features for high-speed batch processing during training epochs.

# Key Technical Decisions
* **Decision 1: MobileNetV2 Architecture (Transfer Learning)**  
  Instead of training a Convolutional Neural Network from scratch, `MobileNetV2` pretrained on ImageNet was utilized. It delivers high accuracy while keeping the model file lightweight (~11 MB), enabling fast inference inside web browsers.
* **Decision 2: Integrated EDA & Evaluation Metrics**  
  Exploratory Data Analysis (`eda.py`) analyzes class distributions via Seaborn/Matplotlib bar charts, while the training script outputs classification reports, accuracy curves, and a saved `confusion_matrix.png` for performance auditing.
* **Decision 3: Gamified Streamlit UI & Local Authentication**  
  Built a fully interactive user interface featuring secure account creation/login via a local database (`users_db.json`), dynamic user rank progression, real-time CO₂ footprint tracking, and point redemption.

# User Interface (UI) Design & Features
As captured in the application preview, the **Streamlit UI** is organized into clean, user-friendly components:

1. **Sidebar Environmental Dashboard (لوحة الأثر البيئي):**
   * **User Greeting & Rank Card:** Dynamically evaluates user points to display personalized titles and encouragement messages (e.g., *🌱 Green Beginner*, *🌿 Active Eco-Activist*, or *🏆 Super Sustainability Champion*).
   * **Live Metrics Trackers:** Real-time metrics counters displaying **Total Points** (e.g., 440 points), **Total Recycled Items** (e.g., 18 items), and **CO₂ Saved** in grams (e.g., 1,360.0g).
   * **Session Control:** Secure logout functionality.

2. **Main Workspace & AI Classification Area:**
   * **File Uploader:** Clean drag-and-drop widget supporting standard image formats (`JPG`, `JPEG`, `PNG`).
   * **Side-by-Side Layout:** Displays the uploaded image on the left and the intelligent analysis results on the right.
   * **Clean Prediction Results:** Displays the identified waste class clearly (e.g., `Plastic`) inside a customized visual container without cluttering the interface with technical confidence percentages.
   * **Smart Ecological Tips:** Instant actionable advice customized to the detected material (e.g., *تأكد من تنظيف العبوات البلاستيكية من السوائل قبل فرزها*).
   * **Interactive Reward Confirmation:** A gradient action button (`♻️ تأكيد إعادة التدوير وكسب النقاط`) that credits user points, updates item counters, triggers celebratory animations (`st.balloons()`), and refreshes the metrics instantly.


# Project Directory Structure
waste/
│
├── app.py                   # Streamlit web application interface & UI logic
├── train.py                 # MobileNetV2 training script & evaluation pipeline
├── eda.py                   # Exploratory Data Analysis & class distribution script
├── waste_classifier_model.h5# Saved TensorFlow trained deep learning model
├── users_db.json            # Local JSON database tracking users, points, and statistics
├── requirements.txt         # Project dependencies and libraries
├── class_distribution.png   # Dataset class breakdown chart
├── confusion_matrix.png     # Model evaluation confusion matrix plot
└── EcoSort_Presentation.pptx# Project presentation slides
