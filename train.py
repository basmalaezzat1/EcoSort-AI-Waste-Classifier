import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. تحديد مسار الداتا
data_dir = r"C:\Users\DELL\OneDrive\Desktop\ODC,Ischool\waste\data\Data\Garbage_Classification"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# 2. تجهيز البيانات وتقسيمها (Train & Validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# 3. تحسين أداء تحميل البيانات
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. إضافة Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# 5. بناء الموديل باستخدام Transfer Learning (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

# 6. تجميع النموذج (Compile)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# 7. بدء التدريب (Training)
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# حفظ الموديل النهائي
model.save("waste_classifier_model.h5")
print("تم حفظ الموديل بنجاح!")


# ==========================================
# 8. قسم التقييم واستخراج Confusion Matrix
# ==========================================
print("\n--- جاري تقييم الموديل واستخراج النتائج للبريزنتيشن ---")

y_true = []
y_pred = []

# جمع كل الصور الحقيقية والتوقعات من الـ validation dataset
for images, labels in val_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# طباعة تقرير الأداء (Classification Report)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# رسم Confusion Matrix وحفظها كصورة للبريزنتيشن
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Waste Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("تم حفظ رسمة الـ Confusion Matrix باسم 'confusion_matrix.png' جاهزة للسلايدز!")