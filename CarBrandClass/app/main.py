# libarry FastAPI ในการสร้าง server
from fastapi import FastAPI, Request

from fastapi.middleware.cors import CORSMiddleware

import requests
# เข้าถึง file code เพื่อใช้งาน Method predictModel
from app.code import predictModel

# สร้าง object ของ FastAPI
app = FastAPI()

# เพิ่ม middleware เพื่อจัดการ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # กำหนดให้ทุกโดเมนสามารถเข้าถึงได้
    allow_methods=['*'],  # กำหนดให้สามารถใช้ทุก HTTP method
    allow_headers=['*']   # กำหนดให้สามารถใช้ทุก header
)

@app.get("/")
def read_root():
    return {"Class Brand Car"}

# 1. สร้าง api เพื่อรอรับ base64 ของภาพ
# 2. ส่ง base64 ที่ได้ ให้กับ 'http://localhost:8080/api/gethog/' จะได้รับค่า HOG มา
# 3. นำค่า HOG ที่ได้ มาเข้า Model เพื่อหาว่า ค่า hog นั้นคือ brand(ยี่ห้อของรถ)
# 4. ส่งกลับ Brand ของรถที่ได้จากการนำเข้า model
@app.post("/api/carbrand") # ใช้ async และ await ใช้คู่กันเพื่อรอรับข้อมูลที่ส่งเข้ามา (ถึงข้อมูลยังไม่มาก็สามารถทำงานต่อได้?)
async def read_image(request: Request): # สร้าง object เพื่อเก็บค่า base64 ที่ส่งมา

    # ส่ง base64 ที่ได้ ให้กับ 'http://localhost:8080/api/gethog/' เพื่อเอาค่า HOG กลับมา
    ########### IPAddress ของ Containers เมื่อต้องการให้ Containers สื่อสารกัน ############
    ########### คำสั่งดู IPAddress ของ Containers ที่ run อยู่ --> docker inspect <รหัสของ Containers นั้น เช่น 9095c139ada5764c7e1460c930ea9f3fb1e26d696061c1e97d7b4a230dde14aa>
    ########### รูปแบบ port เช่น 5050:80 = 5050 คือ port ที่ภานนอกสามารถเรียกใช้งาน Container ตัวนั้นๆ แต่ 80 คือ port ที่ Container ใช้สื่อสารกับ Container ตัวอื่น และ เลข IPAddress ก็ต่างกัน
    ########### เมื่อนำไฟล์ขึ้น docker จะต้องแปลง IPAddress ของเส้นทางที่เราต้องการเรียกใช้ Containers ด้วยกันเอง (หรือก็คือ Containers 2 ตัวคุยกันเอง) เช่น 'http://172.17.0.2:80/api/gethog/'
    path_gethog = 'http://172.17.0.2:80/api/gethog/'

    # รับข้อมูล json มาเป็นแบบ endcode ดังนั้นต้องใช้ .json() เพื่อ decode json ให้เข้าถึงข้อมูลได้
    # ใช้ await เพื่อรอรับข้อมูล ต่อไป
    data = await request.json()

    # เรียกใช้ api โดยส่ง base64 ที่รับมานั้น ไปอีกที
    hog = requests.post(path_gethog, json=data)

    # ค่าที่ตอบ hog กลับมานั้นจะเป็นแบบ json ดังนั้นต้องแปลงให้เป็น list
    # ค่าที่ตอบกลับมา จะมี 2 ค่า คือ HOG Length และ HOG Descriptor แต่เราต้องการแค่ HOG Descriptor
    hog = hog.json()['HOG Descriptor']

    # เรียกใช้ Method predictModel ที่อยู่ในไฟล์ code โดยส่งค่า hog แบบ 2 มิติเข้าไปให้กับ model
    brand =  predictModel([hog])

    return {'Brain is' : brand}