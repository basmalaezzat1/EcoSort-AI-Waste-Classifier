import os
import matplotlib.pyplot as plt
import seaborn as sns

# مسار الداتا الأساسي
data_dir = r"C:\Users\DELL\OneDrive\Desktop\ODC,Ischool\waste\data\Data\Garbage_Classification"

# جلب أسماء الفئات
classes = os.listdir(data_dir)
counts = []

print("--- عدد الصور في كل فئة (Class Distribution) ---")
for c in classes:
    class_path = os.path.join(data_dir, c)
    if os.path.isdir(class_path):
        num_imgs = len(os.listdir(class_path))
        counts.append(num_imgs)
        print(f"- {c}: {num_imgs} صورة")

# رسم Bar Chart لتوضيح الـ Class Imbalance للبريزنتيشن
plt.figure(figsize=(10, 6))
sns.barplot(x=classes, y=counts, palette="viridis")
plt.title("Distribution of Images per Class (Garbage Classification)")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()

# حفظ الصورة عشان تحطيها مباشرة في الـ Presentation
plt.savefig("class_distribution.png")
plt.show()