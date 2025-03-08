from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=200)
    volume = models.FloatField()
    past_sales = models.IntegerField()
    storage_days = models.IntegerField()
    profit = models.FloatField()
    category = models.CharField(max_length=100)
    allocated_rack = models.CharField(max_length=50)

    def __str__(self):
        return self.name

class Transaction(models.Model):
    order_id = models.CharField(max_length=50)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Order {self.order_id} - {self.product.name}"
