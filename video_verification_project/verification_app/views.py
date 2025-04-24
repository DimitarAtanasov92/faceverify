from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import StartVerificationForm
from .models import UserVerification


@login_required
def start_verification(request):
    form = StartVerificationForm()
    return render(request, 'start_verification.html', {'form': form})


@login_required
def verification_process(request):
    if request.method == 'POST':
        # Тук ще стартира процеса на видео верификация
        # Ще пренасочим потребителя към страница с видео стрийминг
        return render(request, 'video_verification.html')
    return redirect('start_verification')


@login_required
def verification_status(request):
    try:
        verification = request.user.userverification
    except UserVerification.DoesNotExist:
        verification = None
    return render(request, 'verification_status.html', {'verification': verification})

