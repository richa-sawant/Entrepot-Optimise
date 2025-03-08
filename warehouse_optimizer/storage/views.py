from django.shortcuts import render
from .models import Product, Transaction
import pandas as pd

def product_list(request):
    products = Product.objects.all()
    return render(request, "storage/product_list.html", {"products": products})

def upload_transactions(request):
    if request.method == "POST" and request.FILES.get("file"):
        df = pd.read_csv(request.FILES["file"])
        for _, row in df.iterrows():
            product, _ = Product.objects.get_or_create(name=row["product_name"])
            Transaction.objects.create(order_id=row["order_id"], product=product)
    return render(request, "storage/upload_transactions.html")
