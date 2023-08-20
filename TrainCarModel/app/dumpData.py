import requests
import cv2
import base64
import os
import numpy as np
import pickle

#======================================================================= สร้าง Method เพื่อเรียกใช้ api แปลง ภาพให้เป็น feature vector

    ########### IPAddress ของ Containers เมื่อต้องการให้ Containers สื่อสารกัน ############
    ########### คำสั่งดู IPAddress ของ Containers ที่ run อยู่ --> docker inspect <รหัสของ Containers นั้น เช่น 9095c139ada5764c7e1460c930ea9f3fb1e26d696061c1e97d7b4a230dde14aa>
    #poth_gethog = 'http://172.17.0.2:8080/api/gethog/'

# api ของ endpoint ถึงเอา feature vector ของรูปรถ โดยส่ง base64 ไป และรับแบบ post
url = 'http://localhost:8080/api/gethog/'

# Method เรียกใช้ api แปลงรูปภาพ เพื่อถึงเอา feature vector ของรูปรถ
def featureVector(img_path):
    # อ่านรูปภาพ จาก path
    img = cv2.imread(img_path)
    # เข้ารหัสภาพให้อยู่ในรูปแบบบิตของ binary ในรูปแบบ .jpg
    retval, buffer = cv2.imencode('.jpg', img)
    # แปลงภาพให้เป็น base64 (ชนิด bytes)
    img_base64 = base64.b64encode(buffer)
    # แปลง base64 ให้เป็น String โดยตัดข้อข้อความตรงจนถึง ' ออก แล้วต่อข้อความใหม่("image data")
    img_base64 = "image data," + str.split(str(img_base64), "'")[1]
    # สร้างชุดข้อมูลที่ต้องการส่ง requests แบบ json {key : value}
    data = {"image_base64": img_base64}
    # เรียก api แบบ post โดยส่ง รูปภาพ แบบ base64 ไป
    response = requests.post(url, json=data)
    # ส่งกลับค่า hog feature vector ที่ได้จากการเรียก api
    return response.json()

#======================================================================= จัดรูปแบบข้อมูลที่ได้จาก dataset สำหรับ train/test model

# สร้าง Method set Train file and Test File เพื่อ เตรียมข้อมูลสำหรับการสร้าง file train/test dataset
def setTrainTestFile(dataPath):

    # สร้าง list x เพื่อเก็บ feature vector ของรูปภาพแต่ละภาพ ที่ได้จากการเรียกใช้ api endpoint
    x = []

    # สร้าง list y เพื่อเก็บ ชื่อ bland ตามจำนวนของ รูปรถ
    y = []

    # สร้าง list mix_xy เพื่อเก็บ list x และ list y รวมกัน
    mix_xy = []

    # os.listdir ส่งกลับชื่อ folder ที่อยู่ใน part นั้น
    Brands_List = os.listdir(dataPath)

    # เข้าถึงยี่ห้อต่างๆ ของรถ
    for brand in Brands_List:
        # os.path.join นำ path ทั้ง 2 มาต่อกันเพื่อให้ได้ path ใหม่    
        brand_path = os.path.join(dataPath, brand)

        # os.listdir ส่งหลับชื่อ file/folder ที่อยู่ใน path นั้นๆ
        cars_List = os.listdir(brand_path)

        # เข้าถึงรูปรถใน ยี่ห้อ นั้นๆ
        for car in cars_List:
            # path รูปภาพของรถนั้น ที่อยู่ใน ยี่ห้อนั้นๆ
            img_file_name = os.path.join(brand_path, car)

            # เพิ่ม path รูปรถเข้าไปใน list x
            x.append(img_file_name)

            # เพิ่มยี่ห้อของรถนั้นๆ เข้าไปใน list y
            y.append(brand)

    #======================================================================= เรียกใช้ Method featureVector แล้วเก็บค่า hog ไว้ใน lsit mix_xy

    # ใช้ enumberate เพื่อเข้าถึง list โดยจะส่ง ตำแหน่ง และค่าที่อยู่ใน ตำแหน่งนั้นๆ
    for index, value in enumerate(x):
        # เรียกใช้ api เพื่อรับค่า hog กลับมา
        hog = featureVector(x[index])

        # ตรวจสอบว่า ได้รับค่า hog กลับมาหรือไม่
        if hog:

            # ค่า hog ที่ได้รับมานั้น จะมี 2 ตัวคือ key ที่ชื่อ HOG Length และ Key ที่ชื่อ HOG Descriptor โดย Key 2 ตัวนี้เก็บค่าต่างกัน แต่สิ่งที่เราต้องการนั้น คือค่า(Value) ที่อยู่ใน Key ที่ชื่อ HOG Descriptor
            # แปลงค่า hog ที่ได้มา ให้เป็น list
            hog = list(hog['HOG Descriptor'])

            # เพิ่มยี่ห้อให้กับค่า hog นั้นๆ ว่าเป็นรูปรถยี่ห้ออะไร (brand จะอยู่ในตำแหน่ง สุดท้ายของ list hog)
            # โดย list y จะเก็บยี่ห้อของรูปภาพรถคันนั้นๆ (x และ y index ตรงกันจากขั้นตอน จัดรูปแบบข้อมูลที่ได้จาก dataset)
            hog.append(y[index])

            # นำค่า hog ที่ได้มา ไปเพิ่มลงใน mix_xy
            mix_xy.append(hog)
        else:
            print('=====HOG Response is Error=====')

    return mix_xy
 
#======================================================================= ทำการบันทึก Feature vector และ Brand(อยู่ index ที่ ตำแหน่งสุกท้ายของ ค่า hog) สำหรับ train ลงไฟในไฟล์นามสกุล .pkl

# dataser path ของชุดข้อมูล
train_path = r'Cars Dataset\train'

# list ค่า hog และ brand(อยู่ในตำแหน่งสุดท้านของค่า hog) สำหรับ train
Train_Cars_Dataset_FeatureVector = setTrainTestFile(train_path)

# กำหนด path หรือ ไฟล์ที่ต้องการสร้าง
train_file_name = 'TrainFeatureVector.pkl'

# นำ list Cars_Dataset_FeatureVector มาสร้างสร้างไฟล์ .pkl
pickle.dump(Train_Cars_Dataset_FeatureVector, open(train_file_name, 'wb'))

print('Finished creating a file called TrainFeatureVector.pkl')

#======================================================================= กำหนดตัวแปล list x_data เพื่อเก็บ path ของรูปภาพรถ และ list y_data เพื่อเก็บ ยี่ห้อของรูปภาพรถคันนั้นๆ และ list Cars_Dataset_FeatureVevtor เพื่อเก็บ feature vector และ Brand(ในตำแหน่งสุดท้ายที่ 8101-1)

# dataser path ของชุดข้อมูล
test_path = r'Cars Dataset\test'

# list ค่า hog และ brand(อยู่ในตำแหน่งสุดท้านของค่า hog) สำหรับ test
Test_Cars_Dataset_FeatureVector = setTrainTestFile(test_path)

# กำหนด path หรือ ไฟล์ที่ต้องการสร้าง
test_file_name = 'TestFeatureVector.pkl'

# นำ list Cars_Dataset_FeatureVector มาสร้างสร้างไฟล์ .pkl
pickle.dump(Test_Cars_Dataset_FeatureVector, open(test_file_name, 'wb'))

print('Finished creating a file called TestFeatureVector.pkl')

