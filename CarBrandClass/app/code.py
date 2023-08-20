# ทดสอบการทำงานของ Model ในไฟล์ ClassifierCarModel.pkl

import pickle
import os
# import json

# ใช้ pickle โหลด Model
# model = pickle.load(open(os.getcwd()+r'model\ClassifierCarModel.pkl', 'rb'))
model = pickle.load(open(os.getcwd()+r'/model/ClassifierCarModel.pkl', 'rb'))

# สร้าง Method เพื่อ predict model โดยการส่งค่า hog เข้าไป predict
def predictModel(hog):
    # ผมลัพท์ brand ของรูปภาพรถ ที่ได้จาก model
    brand = model.predict(hog)
    # คำตอบที่ได้จะเป็น list แต่ใน list นั้นจะมีเพียงตำตอบเดียว จึงส่งกลับตำแหน่งที่ 0
    return brand[0]


# # อ่านไฟล์ json 'r : อ่าน' แล้วเก็บไว้ในตัวแปล json_file
# with open(r'response_data.json', 'r') as json_file:
#     # ใช้ json.load json_file ที่ได้จากการอ่าน
#     data = json.load(json_file)['HOG Descriptor']

# # ส่ง data แบบ 2 มิติ เข้าไปทดสอบใน model
# print(predictModel([data]))
# print(type([data]))