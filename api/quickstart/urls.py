
from django.contrib import admin
from django.urls import path
from .views import YourView, FileUploadView, CheckMalariaView

print("app urls")

urlpatterns = [
    path('songs/', YourView.as_view(), name="songs-all"),
    path('test/', FileUploadView.as_view(), name="fileuploadview"),
    path('check_malaria/', CheckMalariaView.as_view(), name="check_malaria")

]
