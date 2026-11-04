from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.utils import timezone
from django.http import HttpResponse
from django.db import transaction
import os, joblib, math
from datetime import datetime, timedelta
from io import BytesIO
from xhtml2pdf import pisa
from .prediction_engine import generate_5_day_forecast
from .models import PredictionRecord, Stock, HistoricalPrice, ModelVersion, PredictionObservation

def home(request):
    latest_prediction = PredictionRecord.objects.first()
    stock = Stock.objects.filter(ticker='AAPL').first()
    historical_data = []
    freshness_status = "Unknown"
    latest_observation_date = None
    last_sync = None
    
    if stock:
        history = HistoricalPrice.objects.filter(stock=stock).order_by('-date')[:30]
        historical_data = [{'date': h.date.strftime('%Y-%m-%d'), 'close': h.close} for h in reversed(history)]
        if history:
            latest_observation_date = history[0].date
            last_sync = stock.last_data_sync
            today = timezone.now().date()
            days_diff = (today - latest_observation_date).days
            if days_diff <= 3: freshness_status = "Up to date"
            elif days_diff <= 7: freshness_status = "Data delayed"
            else: freshness_status = "Data retrieval failed / Stale"
    
    context = {
        'latest_prediction': latest_prediction, 'stock': stock, 'historical_data': historical_data,
        'total_predictions': PredictionRecord.objects.count(), 'freshness_status': freshness_status,
        'latest_observation_date': latest_observation_date, 'last_sync': last_sync,
    }
    return render(request, 'core/home.html', context)

@login_required
def generate_prediction(request):
    if request.method == 'POST':
        try:
            record, predictions, last_close = generate_5_day_forecast('AAPL')
            messages.success(request, "Successfully generated 5-day forecast!")
        except Exception as e:
            messages.error(request, f"Failed to generate prediction: {str(e)}")
        return redirect('home')
    return redirect('home')

def prediction_history(request):
    records = PredictionRecord.objects.all().order_by('-prediction_date')
    return render(request, 'core/history.html', {'records': records})

def methodology(request):
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_pipeline', 'artifacts')
    model_path = os.path.join(artifacts_dir, 'best_model.joblib')
    features_path = os.path.join(artifacts_dir, 'feature_cols.joblib')
    model_info = {'algorithm': 'Unknown', 'version': '1.0', 'features': [], 'last_training_date': 'Unknown', 'forecast_horizon': 5, 'data_source': 'Yahoo Finance (yfinance)'}
    try:
        if os.path.exists(model_path):
            mtime = os.path.getmtime(model_path)
            model_info['last_training_date'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            model = joblib.load(model_path)
            model_info['algorithm'] = type(model).__name__
        if os.path.exists(features_path):
            features = joblib.load(features_path)
            model_info['features'] = features if isinstance(features, list) else list(features)
    except Exception as e:
        model_info['error'] = str(e)
    return render(request, 'core/methodology.html', {'model_info': model_info})

@login_required
def export_prediction_pdf(request, record_id):
    record = get_object_or_404(PredictionRecord, id=record_id)
    stock = record.stock
    latest_history = HistoricalPrice.objects.filter(stock=stock, date__lt=record.prediction_date).order_by('-date').first()
    context = {
        'record': record, 'stock': stock,
        'latest_observation_date': latest_history.date if latest_history else 'N/A',
        'starting_price': latest_history.close if latest_history else 0.0,
        'generation_date': record.prediction_date.strftime('%Y-%m-%d %H:%M'),
        'model_info': {'algorithm': 'XGBoost / RandomForest', 'version': record.model_version}
    }
    html = render(request, 'core/prediction_report.html', context).content.decode('utf-8')
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), result)
    if not pdf.err:
        response = HttpResponse(result.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="Stock_Forecast_{stock.ticker}_{record.id}.pdf"'
        return response
    return HttpResponse('Error generating PDF', status=500)

@staff_member_required
def model_management(request):
    models_list = ModelVersion.objects.all().order_by('-training_date')
    active_model = ModelVersion.objects.filter(is_active=True).first()
    return render(request, 'core/model_management.html', {'models_list': models_list, 'active_model': active_model})

@staff_member_required
@transaction.atomic
def activate_model(request, model_id):
    model = get_object_or_404(ModelVersion, id=model_id)
    if not os.path.exists(model.artifact_path):
        messages.error(request, f"Cannot activate: Artifact not found at {model.artifact_path}")
        return redirect('model_management')
    ModelVersion.objects.filter(is_active=True).update(is_active=False)
    model.is_active = True
    model.save()
    messages.success(request, f"Successfully activated model version: {model.version_name}")
    return redirect('model_management')

def performance_tracking(request):
    """Evaluates prediction performance against actual market data."""
    today = timezone.now().date()
    
    # 1. Fetch all predictions and ensure observations exist
    records = PredictionRecord.objects.all().order_by('-prediction_date')
    
    total_ae = 0.0
    total_se = 0.0
    total_pe = 0.0
    evaluated_count = 0
    correct_direction = 0
    total_direction_checks = 0
    
    chart_data = [] # For Chart.js
    
    for record in records:
        # Generate observations if missing (backfill for old records)
        if record.observations.count() == 0:
            pred_closes = [record.day_1_close, record.day_2_close, record.day_3_close, record.day_4_close, record.day_5_close]
            current_date = record.prediction_date.date()
            for i in range(5):
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5: current_date += timedelta(days=1)
                PredictionObservation.objects.create(
                    prediction_record=record, forecast_date=current_date, predicted_close=pred_closes[i]
                )
        
        obs_list = record.observations.all()
        record_evaluated_count = 0
        
        for obs in obs_list:
            if obs.forecast_date <= today and not obs.is_evaluated:
                actual = HistoricalPrice.objects.filter(stock=record.stock, date=obs.forecast_date).first()
                if actual:
                    obs.actual_close = actual.close
                    obs.absolute_error = abs(actual.close - obs.predicted_close)
                    obs.percentage_error = (obs.absolute_error / actual.close) * 100 if actual.close != 0 else 0
                    obs.is_evaluated = True
                    obs.save()
            
            if obs.is_evaluated:
                total_ae += obs.absolute_error
                total_se += obs.absolute_error ** 2
                total_pe += obs.percentage_error
                evaluated_count += 1
                record_evaluated_count += 1
                
                # Directional accuracy (compared to previous day's actual)
                prev_actual = HistoricalPrice.objects.filter(stock=record.stock, date__lt=obs.forecast_date).order_by('-date').first()
                if prev_actual:
                    actual_went_up = obs.actual_close > prev_actual.close
                    pred_went_up = obs.predicted_close > prev_actual.close # Approximate direction
                    if actual_went_up == pred_went_up:
                        correct_direction += 1
                    total_direction_checks += 1
        
        # Prepare chart data for the latest evaluated record
        if record_evaluated_count > 0 and not chart_data:
            chart_data = [{'date': o.forecast_date.strftime('%Y-%m-%d'), 'predicted': o.predicted_close, 'actual': o.actual_close} for o in obs_list if o.is_evaluated]

    mae = total_ae / evaluated_count if evaluated_count > 0 else 0
    rmse = math.sqrt(total_se / evaluated_count) if evaluated_count > 0 else 0
    avg_pe = total_pe / evaluated_count if evaluated_count > 0 else 0
    dir_accuracy = (correct_direction / total_direction_checks) * 100 if total_direction_checks > 0 else 0
    
    context = {
        'mae': round(mae, 2), 'rmse': round(rmse, 2), 'avg_pe': round(avg_pe, 2),
        'dir_accuracy': round(dir_accuracy, 2), 'evaluated_count': evaluated_count,
        'chart_data': chart_data
    }
    return render(request, 'core/performance.html', context)

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                return redirect('home')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'core/login.html', {'form': form})

def user_logout(request):
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('home')
