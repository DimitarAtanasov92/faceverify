import json
import base64
import cv2
import numpy as np
import pytesseract
from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth import get_user_model
from .models import UserVerification

User = get_user_model()

# Конфигуриране на пътя към Tesseract, ако е необходимо
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class VerificationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope['user']
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        image_type = text_data_json.get('type')
        image_data_url = text_data_json.get('image')

        if image_data_url:
            try:
                format, imgstr = image_data_url.split(';base64,')
                decoded_image = base64.b64decode(imgstr)
                nparr = np.frombuffer(decoded_image, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image_type == 'face':
                    await self.process_face_frame(img)
                elif image_type == 'document_front':
                    await self.process_document(img, 'front')
                elif image_type == 'document_back':
                    await self.process_document(img, 'back')
            except Exception as e:
                print(f"Грешка при декодиране на изображение: {e}")
                await self.send(text_data=json.dumps({'error': 'Грешка при обработка на изображение'}))

    async def process_face_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('verification_app/haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_detected = len(faces) > 0

        # Опростена логика за "жив човек" (както преди)
        if not hasattr(self, 'previous_face_frame'):
            self.previous_face_frame = gray
            liveness_detected = False
        else:
            frame_diff = cv2.absdiff(gray, self.previous_face_frame)
            mean_diff = np.mean(frame_diff)
            liveness_detected = mean_diff > 5
            self.previous_face_frame = gray

        await self.send(text_data=json.dumps({
            'face_detected': face_detected,
            'liveness_detected': liveness_detected
        }))

        if face_detected and liveness_detected:
            await self.update_verification_status(self.user, {'face_verified': True, 'liveness_verified': True})
            await self.send(text_data=json.dumps({'verification_status': 'Лицето и жив човек са потвърдени.'}))

    async def process_document(self, img, side):
        try:
            # Извършване на OCR с български език
            text = pytesseract.image_to_string(img, lang='bul')
            print(f"OCR текст ({side}):\n{text}")

            # Тук трябва да се добави логика за извличане на конкретни данни
            # от разпознатия текст (например с регулярни изрази)
            extracted_data = self.extract_document_data(text)

            if side == 'front':
                await self.update_verification_status(self.user, {'document_front_data': extracted_data})
                await self.send(text_data=json.dumps({'document_data_front': extracted_data}))
            elif side == 'back':
                await self.update_verification_status(self.user, {'document_back_data': extracted_data})
                await self.send(text_data=json.dumps({'document_data_back': extracted_data}))

        except pytesseract.TesseractError as e:
            print(f"Грешка при OCR ({side}): {e}")
            await self.send(text_data=json.dumps({'error': f'Грешка при OCR ({side}): {e}'}))
        except Exception as e:
            print(f"Грешка при обработка на документ ({side}): {e}")
            await self.send(text_data=json.dumps({'error': f'Грешка при обработка на документ ({side}): {e}'}))

    def extract_document_data(self, text):
        # *** Много опростена логика за извличане на данни ***
        # В реално приложение ще се използват по-сложни методи като
        # регулярни изрази, анализ на ключови думи и структура на документа.
        data = {}
        lines = text.split('\n')
        for line in lines:
            if "Име" in line:
                parts = line.split("Име")
                if len(parts) > 1:
                    data['име'] = parts[1].strip()
            elif "Фамилия" in line:
                parts = line.split("Фамилия")
                if len(parts) > 1:
                    data['фамилия'] = parts[1].strip()
            elif "Дата на раждане" in line:
                parts = line.split("Дата на раждане")
                if len(parts) > 1:
                    data['дата_раждане'] = parts[1].strip()
            # ... добавете други полета
        return data

    async def update_verification_status(self, user, data):
        try:
            verification, created = await UserVerification.objects.aget_or_create(user=user)
            if 'face_verified' in data:
                verification.лицева_верификация_успешна = data['face_verified']
            if 'liveness_verified' in data:
                verification.жив_човек_верификация_успешна = data['liveness_verified']
            if 'document_front_data' in data:
                verification.сканирани_данни_предна_страна = data['document_front_data']
            if 'document_back_data' in data:
                verification.сканирани_данни_задна_страна = data['document_back_data']
            await verification.asave()
        except Exception as e:
            print(f"Грешка при запис на статус на верификация: {e}")