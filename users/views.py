from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views # За логин/логаут
from .forms import CustomUserCreationForm
from verification.models import VerificationDocument # За проверка на статус

class RegisterView(CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('login') # Пренасочва към логин след успешна регистрация
    template_name = 'users/registration/register.html'

# Използваме вградените изгледи за логин и логаут, но можем да ги персонализираме
class CustomLoginView(auth_views.LoginView):
    template_name = 'users/registration/login.html'

class CustomLogoutView(auth_views.LogoutView):
    next_page = reverse_lazy('login') # Пренасочва към логин след изход

@login_required
def profile_view(request):
    user = request.user
    verification_doc = None
    try:
        # Опитваме да вземем документа за верификация, ако съществува
        verification_doc = VerificationDocument.objects.get(user=user)
    except VerificationDocument.DoesNotExist:
        pass # Няма проблем, потребителят още не е стартирал процеса

    context = {
        'user': user,
        'profile': user.profile,
        'verification_doc': verification_doc,
    }
    return render(request, 'users/profile.html', context)