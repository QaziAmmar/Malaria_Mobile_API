from django.shortcuts import render

# Create your views here.
from rest_framework import views
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser
from .serializers import YourSerializer, FileSerializer
from rest_framework import status
from rest_framework.views import APIView
from .cv_iml import *


class YourView(views.APIView):

    def post(self, request):
        print(request.data['title'])
        yourdata = [{"likes": "10", "comments": "0"}, {"likes": "4", "comments": "23"}]
        results = YourSerializer(yourdata, many=True).data
        return Response(results)


class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):

        file_serializer = FileSerializer(data=request.data)
        # print("file_serializer", file_serializer)
        # image = request.data['file']
        # title = request.data['title']
        if file_serializer.is_valid():
            file_serializer.save()
            image = request.data['file']
            imagepath = "/Users/qaziammar/Documents/Pycharm/DjanogPractice/api/" + str(image)
            yourdata = red_blood_cell_segmentation(imagepath)
            data = {"imagePoints": yourdata}
            returnData = {"data": data}
            # results = YourSerializer(yourdata, many=True).data
            # print(len(yourdata))

            os.remove(imagepath)
            return Response(returnData, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CheckMalariaView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):

        file_serializer = FileSerializer(data=request.data)
        # print("file_serializer", file_serializer)
        # image = request.data['file']
        # title = request.data['title']
        if file_serializer.is_valid():
            file_serializer.save()
            image = request.data['file']
            imagepath = "/Users/qaziammar/Documents/Pycharm/DjanogPractice/api/" + str(image)
            prediction = check_image_malaria(imagepath)
            returnData = {"data": prediction}
            # results = YourSerializer(yourdata, many=True).data
            # print(len(yourdata))

            return Response(returnData, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)