import cv2
import pytesseract
import base64
import numpy as np
import json

from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from .forms import UserProfileForm
from .models import UserProfile


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.create(user=user)
            return redirect('verify_user')
    else:
        form = UserCreationForm()
    return render(request, 'verification/register.html', {'form': form})


def verify_user(request):
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES)
        if form.is_valid():
            user_profile = form.save(commit=False)
            user_profile.user = request.user
            user_profile.save()
            return redirect('verification_success')
    else:
        form = UserProfileForm()
    return render(request, 'verification/verify.html', {'form': form})


@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data['image']
        name = data['name']

        # Декодиране на изображението
        image_data = image_data.split(',')[1]  # Премахване на "data:image/png;base64,"
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)

        # Определяне на изображението
        if name == 'id_image':
            # Обработка на ID изображението
            id_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            extracted_data = extract_data_from_image(id_image)
            return JsonResponse({'status': 'success', 'data': extracted_data})

        elif name == 'face_image':
            # Обработка на лице
            face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Тук можеш да добавиш логика за разпознаване на лицето
            return JsonResponse({'status': 'success', 'message': 'Лицето е заснето.'})

    return JsonResponse({'status': 'error', 'message': 'Невалидно искане.'})


def extract_data_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text


def verification_success(request):
    return render(request, 'verification/success.html')