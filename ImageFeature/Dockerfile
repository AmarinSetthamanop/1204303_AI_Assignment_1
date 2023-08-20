# FROM python:3.9

# # 
# WORKDIR /code

# # 
# COPY ./requirements.txt /code/requirements.txt

# # 
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# # 
# COPY ./app /code/app

# # CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# # หากคุณใช้งานคอนเทนเนอร์ของคุณหลัง TLS Termination Proxy (โหลดบาลานเซอร์)
# # เช่น Nginx หรือ Traefik ให้เพิ่มตัวเลือก--proxy-headersซึ่งจะเป็นการบอก Uvicorn
# # ให้เชื่อถือส่วนหัวที่ส่งโดยพร็อกซีนั้นโดยบอกว่าแอปพลิเคชันทำงานหลัง HTTPS เป็นต้น
# CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]


FROM python:3.9

WORKDIR /ImageHog

COPY ./requirements.txt /ImageHog/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /ImageHog/requirements.txt

COPY ./app /ImageHog/app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
ENV PYTHONPATH = "${PYTHONPATH}:/ImageHog"

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80" ]