================================================================
รายละเอียดการพัฒนาโมเดล — Project IS 2568
================================================================

================================================================
โมเดลที่ 1: Machine Learning (Ensemble)
Dataset: Vehicle Type Recognition
================================================================

--- ที่มาของ Dataset ---
ดาวน์โหลดจาก Kaggle — Vehicle Type Recognition Dataset
https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification

--- Feature ของ Dataset ---
ภาพยานพาหนะ 7 ประเภท (Auto Rickshaws, Bikes, Cars, Motorcycles,
Planes, Ships, Trains) ถ่ายจากมุมมองและพื้นหลังหลากหลาย
รูปแบบไฟล์ JPEG/PNG ขนาดภาพไม่เท่ากัน
จำนวนรูปภาพทั้งหมด: 5,590 ภาพ

--- ความไม่สมบูรณ์ของ Dataset ---
- ขนาดภาพไม่เท่ากันในแต่ละไฟล์
- บางไฟล์ corrupted (cv2.imread คืนค่า None)
- จำนวนภาพ Cars น้อยกว่า class อื่น (790 vs 800 ภาพ)

--- การเตรียมข้อมูล (Data Preprocessing) ---
1. Load & Filter: อ่านภาพด้วย cv2.imread() และกรองภาพ
   corrupted ออกด้วย if img is None: continue
2. Resize: ปรับขนาดทุกภาพเป็น 128×128 px ด้วย cv2.resize()
   เพื่อให้ได้ feature vector ขนาดสม่ำเสมอ
3. Label Encoding: แปลง class name (string) เป็นตัวเลข
   ด้วย LabelEncoder
4. HOG Feature Extraction: แปลง RGB → Grayscale แล้วสกัด
   HOG feature ด้วยพารามิเตอร์
   - orientations=12
   - pixels_per_cell=(8, 8)
   - cells_per_block=(2, 2)
   - block_norm='L2-Hys'
   ได้ feature vector shape (5590, 10800)
5. Train/Test Split: แบ่ง 80:20 ด้วย stratify=y, random_state=42
   → Train: 4,472 | Test: 1,118
6. StandardScaler: fit_transform() บน train เท่านั้น
   แล้ว transform() บน test เพื่อป้องกัน data leakage
7. PCA: ลด dimension ด้วย PCA(n_components=0.95)
   จาก 10,800 → 1,979 features (คง 95% ของ variance)

--- ทฤษฎีของอัลกอริทึม ---

HOG (Histogram of Oriented Gradients):
  สกัด feature โดยวิเคราะห์ทิศทางและความแรงของ gradient
  ในแต่ละ cell แทนการใช้ pixel value โดยตรง ทำให้ทนทาน
  ต่อการเปลี่ยนแปลงของแสงและสีได้ดี

SVM (Support Vector Machine — RBF Kernel):
  หาระนาบ (Hyperplane) ที่แบ่งข้อมูลระหว่าง class
  โดยให้ margin กว้างที่สุด RBF Kernel แปลงข้อมูลเข้าสู่
  high-dimensional space เพื่อแบ่ง non-linear data ได้
  พารามิเตอร์: kernel='rbf', C=10, gamma='scale',
  probability=True
  Accuracy: 80.9%

Random Forest:
  สร้าง Decision Tree จำนวน 500 ต้น แต่ละต้นฝึกบน
  subset ของข้อมูลแบบสุ่ม (Bootstrap Sampling) ผลลัพธ์
  ได้จาก majority vote ของทุก tree
  พารามิเตอร์: n_estimators=500, max_depth=None,
  min_samples_split=2, n_jobs=-1
  Accuracy: 71.6%

XGBoost (Extreme Gradient Boosting):
  สร้าง tree แบบ sequential โดยแต่ละต้นเรียนรู้จาก
  residual error ของต้นก่อนหน้า มีกลไก regularization
  ผ่าน subsample และ colsample_bytree
  พารามิเตอร์: n_estimators=400, max_depth=6,
  learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
  Accuracy: 78.4%

Soft Voting Classifier (Ensemble):
  รวม 3 โมเดลโดยให้แต่ละโมเดล output ความน่าจะเป็นของ
  ทุก class จากนั้นเฉลี่ย probability ทั้ง 3 โมเดล แล้ว
  predict class ที่มี probability รวมสูงสุด
  Ensemble Accuracy: 82.3%

--- ขั้นตอนการพัฒนาโมเดล ---
Image → Resize 128×128 → Grayscale → HOG (10,800 features)
→ StandardScaler → PCA (1,979 features)
→ SVM + Random Forest + XGBoost
→ Soft Voting Classifier → Prediction

--- ผลการทดสอบ (Per-Class Performance) ---
Class            Precision  Recall  F1-Score  Support
Auto Rickshaws   0.83       0.78    0.80      160
Bikes            0.99       0.93    0.95      160
Cars             0.86       0.73    0.79      158
Motorcycles      0.87       0.81    0.84      160
Planes           0.86       0.81    0.84      160
Ships            0.64       0.93    0.76      160
Trains           0.82       0.78    0.80      160
Overall Accuracy: 82.3%

--- แหล่งอ้างอิง ---
[1] Dataset: Vehicle Type Recognition Dataset. Kaggle.
    https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification
[2] HOG: Dalal, N., & Triggs, B. (2005). Histograms of oriented
    gradients for human detection. IEEE CVPR 2005, 886-893.
    doi:10.1109/CVPR.2005.177
[3] SVM: Cortes, C., & Vapnik, V. (1995). Support-vector networks.
    Machine Learning, 20(3), 273-297.
[4] Random Forest: Breiman, L. (2001). Random Forests.
    Machine Learning, 45(1), 5-32.
[5] XGBoost: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable
    Tree Boosting System. KDD '16, 785-794.
    https://arxiv.org/abs/1603.02754
[6] scikit-learn: Pedregosa et al. (2011). Scikit-learn: Machine
    Learning in Python. JMLR, 12, 2825-2830.
    https://scikit-learn.org


================================================================
โมเดลที่ 2: Neural Network (Transfer Learning)
Dataset: Intel Image Classification
================================================================

--- ที่มาของ Dataset ---
ดาวน์โหลดจาก Kaggle — Intel Image Classification
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

--- Feature ของ Dataset ---
ภาพฉากธรรมชาติและสิ่งปลูกสร้าง 6 ประเภท (Buildings, Forest,
Glacier, Mountain, Sea, Street) ถ่ายในสภาพแสงและสภาพอากาศ
หลากหลาย ขนาดภาพ original 150×150 px
จำนวนภาพ Train: ~14,034 | Test: ~3,000

--- ความไม่สมบูรณ์ของ Dataset ---
- จำนวนภาพใน train ไม่เท่ากันระหว่าง class (2,191–2,512 ภาพ)
- บางภาพมี noise จากสภาพอากาศ (หมอก, ฝน)

--- การเตรียมข้อมูล (Data Preprocessing) ---
1. Load via ImageDataGenerator: โหลดภาพผ่าน
   flow_from_directory() แบ่ง class จาก folder structure
   อัตโนมัติ กำหนด class_mode='categorical'
2. Resize: ปรับขนาดเป็น 160×160 px ให้ตรงกับ input ของ
   MobileNetV2
3. MobileNetV2 preprocess_input: ใช้ preprocessing_function=
   preprocess_input จาก mobilenet_v2 แทน rescale=1/255
   แปลง pixel จาก [0–255] → [-1, 1] ให้ตรงกับที่ MobileNetV2
   ถูก pretrain มา
4. Data Augmentation (train set only):
   - Horizontal Flip
   - Rotation ±15°
   - Zoom 15%
   - Width/Height Shift 10%
   เพื่อเพิ่มความหลากหลายและป้องกัน overfitting
5. Validation Split: แบ่ง 80:20 จาก train data
   → Train: 11,230 | Validation: 2,804

--- ทฤษฎีของอัลกอริทึม ---

Transfer Learning:
  นำน้ำหนักที่ฝึกไว้แล้วบน ImageNet (1.2 ล้านภาพ 1,000 class)
  มาใช้เป็นจุดเริ่มต้น โมเดลได้เรียนรู้ feature ทั่วไป เช่น
  edge, texture, shape ไว้แล้ว ทำให้ใช้ข้อมูลน้อยกว่า
  เทรนเร็วกว่า และ accuracy สูงกว่าการฝึกจาก scratch

MobileNetV2 (Backbone):
  ออกแบบโดย Google สำหรับ mobile device ใช้ Depthwise
  Separable Convolution แยก spatial filtering และ channel
  combination ออกจากกัน ลด computation ลง ~8-9x เทียบกับ
  standard convolution มี Inverted Residual Blocks ที่
  expand channel (×6) → depthwise conv → compress กลับ
  พร้อม shortcut connection เพื่อ preserve gradient flow

โครงสร้าง Model (Custom Head):
  MobileNetV2 (weights='imagenet', include_top=False,
               input=(160,160,3), frozen)
  → GlobalAveragePooling2D()   # ลด spatial → 1 vector
  → BatchNormalization()       # normalize ก่อนเข้า Dense
  → Dense(256, activation='relu')
  → Dropout(0.5)               # ป้องกัน overfitting
  → Dense(6, activation='softmax')
  Total params: 2,922,054

Loss Function: CategoricalCrossentropy(label_smoothing=0.1)

--- ขั้นตอนการพัฒนาโมเดล (2-Phase Training) ---

Phase 1 — Head Training (max 10 epochs):
  Freeze ทุก layer ของ MobileNetV2 เทรนเฉพาะ
  classification head ใหม่
  - Optimizer: Adam(lr=1e-3)
  - EarlyStopping(patience=3, restore_best_weights=True)
  - ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6)

Phase 2 — Fine-tuning (max 10 epochs):
  Unfreeze 40 layer สุดท้ายของ backbone แล้วเทรนต่อ
  ด้วย learning rate ต่ำมากเพื่อป้องกัน catastrophic
  forgetting ของ pretrained weights
  - Optimizer: Adam(lr=5e-6)
  - EarlyStopping และ ReduceLROnPlateau เหมือน Phase 1

Pipeline:
  Image → Resize 160×160 → preprocess_input
  → MobileNetV2 (frozen) → GAP → BatchNorm
  → Dense(256) → Dropout(0.5) → Softmax(6)

--- ผลการทดสอบ ---
Test Accuracy: 91.2%

--- แหล่งอ้างอิง ---
[1] Dataset: Intel Image Classification. Kaggle.
    https://www.kaggle.com/datasets/puneet6060/intel-image-classification
[2] MobileNetV2: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A.,
    & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and
    Linear Bottlenecks. IEEE CVPR 2018.
    https://arxiv.org/abs/1801.04381
[3] Transfer Learning: Pan, S. J., & Yang, Q. (2010). A Survey on
    Transfer Learning. IEEE Transactions on Knowledge and Data
    Engineering, 22(10), 1345-1359.
[4] ImageNet: Deng, J., et al. (2009). ImageNet: A Large-Scale
    Hierarchical Image Database. IEEE CVPR 2009.
[5] Keras / TensorFlow: Chollet, F., et al. (2015). Keras.
    https://keras.io

================================================================