from rest_framework import serializers
from .models import File


class YourSerializer(serializers.Serializer):
    """Your data serializer, define your fields here."""
    comments = serializers.CharField()
    likes = serializers.CharField()


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = File
        fields = "__all__"
