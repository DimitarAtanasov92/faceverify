from django.db import models
from django.contrib.auth.models import User

class UserVerification(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    лицева_верификация_успешна = models.BooleanField(default=False)
    документ_верификация_успешна = models.BooleanField(default=False)
    жив_човек_верификация_успешна = models.BooleanField(default=False)
    сканирани_данни_предна_страна = models.JSONField(null=True, blank=True)
    сканирани_данни_задна_страна = models.JSONField(null=True, blank=True)
    време_на_верификация = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Верификация на {self.user.username}"
