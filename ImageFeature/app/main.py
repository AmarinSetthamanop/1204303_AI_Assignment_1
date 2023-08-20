# ใน main.py ทําการสร้าง service endpoint /api/genhog โดย endpoint นี้จะรับภาพ (base64)
# และประมวลผลเพื่อหาเอกลักษณ์ของภาพด้วย HOG และส่งค่า HOG (vector) กลับไปยังผู้เรียกใช้งาน
# ตัวอย่างของการเขียนเพื่อสกัดเอกลักษณ์ HOG จากภาพ แสดงดังตัวอย่างด้านล่าง

# libarry  fastapi ในการเปิด port api
from fastapi import FastAPI, Response
# libarry สำหรับ Request Body
from pydantic import BaseModel
# libarry แปลง base64
import base64

import cv2
import numpy as np

# สร้าง class เพื่อแปลงจาก json ข้อมูลมาเป็น object
class ImageClass(BaseModel):
    image_base64: str

# Method แปลง base64 ให้เป็น array ของรูปภาพ
def readb64(uri):
    # ตัดส่วนนี้ (data:image/jpeg;base64,) ของ base64 ออก
    encoded_data = uri.split(',')[1]
    # decode base64 ให้ออกมาเป็น byte
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    # ใช้ cv2 อ่านข้อมูลหลังจาก decode base64 แล้ว
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    return img


# สร้าง object FastAPI
app = FastAPI()

# api เริ่มต้น แบบ get
@app.get("/")
def read_root():
    return {"Wellcome to AMARIN api"}

# api แบบ get ที่รอรับค่าต่างๆ เมื่อส่งผ่านมาทาง URL
@app.get("/api/{input_values}")
def read_values(input_values):
    return {"is to" : input_values}

# api รับรูปภาพแบบ base64 แล้วส่งกลับคืน เอกลักษณ์ HOG จากภาพ
@app.post("/api/gethog/")
async def read_image(image: ImageClass):

    # image ที่ได้หลังจากการแปลง Base64 แล้ว
    img =  readb64(image.image_base64)

    # เปรี่ยนขนาดของรูปภาพ
    img = cv2.resize(img, (128, 128), cv2.INTER_AREA)

    # ค่าความกว้างและความสูงของภาพ ต้องอยู่ในขอบเขตที่ cv2 กำหนด จึงต้องทำการ resize img
    win_size = img.shape
    # ค่า block_size ควรเล็กกว่า win_size โดยทั่วไปเพื่อให้คุณสามารถครอบคลุมพื้นที่ภายในหน้าต่างหลักโดยการใช้ช่องส่วนย่อย (block) หลาย ๆ ช่อง
    block_size = (16, 16)
    # ค่า block_stride กำหนดว่าช่องส่วนย่อยจะถูกเคลื่อนย้ายแบบไหนในแต่ละขั้น ในที่นี้คือ 8x8 ซึ่งเป็นค่ามาตรฐาน
    block_stride = (8, 8)
    # cell_size กำหนดขนาดของเซลล์ที่ใช้ในการสร้าง histogram สำหรับแต่ละช่องส่วนย่อย
    cell_size = (8, 8)
    # num_bins กำหนดจำนวน bin ใน histogram ซึ่งส่งผลต่อการแบ่งความต่างของทิศทางของกระแสภาพในแต่ละช่อง
    num_bins = 9

    # ตั้งค่าพารามิเตอร์ของตัวสร้าง HOG โดยใช้ตัวแปรที่กำหนดไว้ข้างต้น
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

    # เพื่อสกัดเอกลักษณ์ HOG จากภาพที่กำหนด และค่าที่คืนมาจะเป็นเวกเตอร์ที่เกี่ยวข้องกับเอกลักษณ์ HOG ของภาพนั้นๆ
    hog_descriptor = hog.compute(img)

    return {"HOG Length" : len(hog_descriptor),
            "HOG Descriptor" : hog_descriptor.tolist()
            }

