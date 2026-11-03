from django.db import models
from django.conf import settings

class Stock(models.Model):
    ticker = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    exchange = models.CharField(max_length=50)
    currency = models.CharField(max_length=10, default='USD')
    last_data_sync = models.DateTimeField(null=True, blank=True, help_text="When this stock's data was last successfully retrieved")

    def __str__(self):
        return f"{self.ticker} - {self.name}"

class HistoricalPrice(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='historical_prices')
    date = models.DateField(unique=True)
    open_price = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.stock.ticker} on {self.date}: ${self.close:.2f}"

class PredictionRecord(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='predictions')
    prediction_date = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50)
    day_1_close = models.FloatField()
    day_2_close = models.FloatField()
    day_3_close = models.FloatField()
    day_4_close = models.FloatField()
    day_5_close = models.FloatField()
    mae_at_time = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-prediction_date']

    def __str__(self):
        return f"Prediction for {self.stock.ticker} on {self.prediction_date.strftime('%Y-%m-%d %H:%M')}"
