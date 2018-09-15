from django.db import models
from django.contrib.auth.models import AbstractUser


# Create your models here.
class User(AbstractUser):
    faceData = models.TextField(default='')
    emotionData = models.TextField(default='{"happy": 0, "neutral": 0, "sad": 0, "angry": 0, "surprise": 0, "disgust": 0, "fear": 0}')
    emotionCount = models.IntegerField(default=0)
    historyAlert = models.TextField(default='')
    underThreshold = models.TextField(default='')
    threshold = models.FloatField(default=0.5)
    update = models.BooleanField(default=False)

    def __str__(self):
        return self.username
