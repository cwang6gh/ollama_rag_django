from django.urls import path
from .views import AskImageView

urlpatterns = [
    path("ask-images/", AskImageView.as_view(), name="ask_images"),
]

