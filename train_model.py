# ============================================================
#   PLANT DISEASE DETECTOR  -  Training Script
#   Hum sirf WHEAT (gandum) se shuru kar rahe hain - 3 classes
# ============================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # extra warning messages chhupane ke liye

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------- SETTING: yahan apne Wheat folder ka path daalein ----------
# Agar aap ne Wheat folder ko project ke andar "dataset" folder mein rakha hai,
# to neeche wali line waisi hi rehne dein:
DATA_DIR = r"dataset\Wheat"
# Ya phir poora path daal sakte hain, misaal:
# DATA_DIR = r"C:\Users\Faraz\Desktop\plant_detector\dataset\Wheat"

IMG_SIZE = (224, 224)

print("1) Images parhi ja rahi hain...")
ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=32,
    shuffle=False,
    label_mode="int",
)
class_names = ds.class_names
print("   Bemariyan (classes) milin:", class_names)

print("2) Pretrained model load ho raha hai (PEHLI BAAR internet chahiye, ek baar download hoga)...")
feature_extractor = MobileNetV2(
    weights="imagenet", include_top=False, pooling="avg",
    input_shape=(224, 224, 3),
)

print("3) Har image se features nikaale ja rahe hain (CPU par thora waqt lagega, sabar karein)...")
X, y = [], []
for images, labels in ds:
    images = preprocess_input(images)
    feats = feature_extractor.predict(images, verbose=0)
    X.append(feats)
    y.append(labels.numpy())
X = np.concatenate(X)
y = np.concatenate(y)
print("   Total images:", X.shape[0])

print("4) Data ko train aur test mein baant rahe hain...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("5) Classifier train ho raha hai...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

print("6) Nataij (results):")
preds = clf.predict(X_test)
print("   Accuracy:", round(accuracy_score(y_test, preds), 3))
print(classification_report(y_test, preds, target_names=class_names))

print("7) Model save ho raha hai...")
joblib.dump(clf, "plant_model.pkl")
joblib.dump(class_names, "class_names.pkl")
print("\nHO GAYA! plant_model.pkl aur class_names.pkl ban gaye hain.")
