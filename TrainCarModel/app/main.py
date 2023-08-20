import pickle
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#======================================================================= อ่านไฟล์ dataset ที่เก็บ feature vector และ Brand(ในตำแหน่ง 8100-1) ของรูปภาพ

train_path = r"TrainFeatureVector.pkl"
test_path = r"TestFeatureVector.pkl"

# อ่านไฟล์ .pkl แล้วมาเก็บในตัวแปลงแบบ list
Train_Cars_Dataset_FeatureVector = pickle.load(open(train_path, 'rb'))
Test_Cars_Dataset_FeatureVector = pickle.load(open(test_path, 'rb'))

#======================================================================= จักรูปแบบข้อมูล x = feature vector และ y = Brand ของรถ สำหรับ train

# สร้าง Method เพื่อ set ข้อมูลให้กับ x_train, y_train, x_test, y_test
def setData(dataset):
    x = []
    y = []
    # กำหนดค่าที่อยู่ใน dataset(2 มิติ) ให้กับ x_train และ y_train
    # ใช้ enumberate เพื่อเข้าถึง list โดยจะส่ง ตำแหน่ง และค่าที่อยู่ใน ตำแหน่งนั้นๆ
    for index, value in enumerate(dataset):

        # เอาข้อมูลที่อยู่ใน dataset [ในตำแหน่งนั้นๆ] [ ในตำแหน่ง x ที่ 0 : จนถึงตัวสุดท้าย ] และลบ(-1)ไปอีก 1 ตำแหน่งเพราะนับ 0 ด้วย
        x.append( dataset[index][ : len(dataset[index])-1] )

        # เอาข้อมูลที่อยู่ใน dataset [ในตำแหน่งนั้นๆ] [ตำแหน่งสุดท้าย - 1 เพราะนับ 0 ด้วย]
        y.append( dataset[index][len(dataset[index])-1] )

    # ส่งกลับ feature vector(x) และ brand(y) ของ feature นั้นๆ
    return x, y

#=======================================================================

# เก็บข้อมูล feature vector(x) และ brand(y) ของ feature นั้นๆ จากการเรียกใช้ Method setData
x_train, y_train = setData(Train_Cars_Dataset_FeatureVector)
x_test, y_test = setData(Test_Cars_Dataset_FeatureVector)

print('x_train:', len(x_train))
print('x_test:', len(x_test))
print('y_train:', len(y_train))
print('y_test:', len(y_test))

#======================================================================= train model หรือ สร้างต้นไม่ตัดสินใจ

# ส้ราง object ของ DecisionTreeClasssifier()
model = DecisionTreeClassifier()
# ส่งช้อมูล x_train, y_train เข้าไปทดสอบ
model = model.fit(x_train, y_train)

# ทดสอบประสิทธิภาพจากชุดข้อมูล โดยส่ง x_test เข้าไป
Ypred = model.predict(x_test)
# ค่าประสิทธิภาพที่ได้ โดยส่ง Ypred, y_test เข้าไป
accuracy = metrics.accuracy_score(y_test, Ypred) * 100

# matrix จากการ test model คำตอบที่ถูกต้องจะอยู่ในรูปแบบ แนวทแยง
matrix = confusion_matrix(y_test, Ypred)

print("\nAccuracy:", accuracy)

print("Confusion matrix:\n", matrix)

#======================================================================= save model ไว้แบบไฟล์

# สร้าง path/file name
file_name = 'ClassifierCarModel.pkl'
# ใช้ pickle เพื่อสร้างไฟล์ model
pickle.dump(model, open(file_name, 'wb'))

print("\nClassifierCarModel.pkl file saved.")

