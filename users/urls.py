from django.urls import path
from .views import RegisterView, CustomLoginView, CustomLogoutView, profile_view

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', CustomLogoutView.as_view(), name='logout'),
    path('profile/', profile_view, name='profile'),
     # Може да сложим профила като default за логнат потребител
    path('', profile_view, name='home'),
]