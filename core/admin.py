from django.contrib import admin
from .models import Stock, HistoricalPrice, PredictionRecord

# Polish Admin Site Headers
admin.site.site_header = "StockPredictor Administration"
admin.site.site_title = "StockPredictor Admin Portal"
admin.site.index_title = "Welcome to the StockPredictor Admin Portal"

@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'name', 'exchange', 'currency')
    search_fields = ('ticker', 'name')

@admin.register(HistoricalPrice)
class HistoricalPriceAdmin(admin.ModelAdmin):
    list_display = ('stock', 'date', 'open_price', 'high', 'low', 'close', 'volume')
    list_filter = ('stock', 'date')
    search_fields = ('stock__ticker',)
    ordering = ('-date',)

@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ('stock', 'prediction_date', 'model_version', 'day_1_close', 'day_2_close', 'day_3_close', 'day_4_close', 'day_5_close')
    list_filter = ('stock', 'model_version', 'prediction_date')
    readonly_fields = ('prediction_date',)
