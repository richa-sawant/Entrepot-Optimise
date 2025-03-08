from django.urls import path
from .views import product_list, upload_transactions

urlpatterns = [
    path("products/", product_list, name="product_list"),
    path("upload/", upload_transactions, name="upload_transactions"),
]
