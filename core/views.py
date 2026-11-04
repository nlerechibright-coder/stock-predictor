from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.utils import timezone
from django.http import HttpResponse
from django.db import transaction
import os
import joblib
from datetime import datetime
from io import BytesIO
from xhtml2pdf import pisa
from .prediction_engine import generate_5_day_forecast
from .models import PredictionRecord, Stock, HistoricalPrice, ModelVersion

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
            
            if days_diff <= 3:
                freshness_status = "Up to date"
            elif days_diff <= 7:
                freshness_status = "Data delayed"
            else:
                freshness_status = "Data retrieval failed / Stale"
    
    context = {
        'latest_prediction': latest_prediction,
        'stock': stock,
        'historical_data': historical_data,
        'total_predictions': PredictionRecord.objects.count(),
        'freshness_status': freshness_status,
        'latest_observation_date': latest_observation_date,
        'last_sync': last_sync,
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
    
    model_info = {
        'algorithm': 'Unknown',
        'version': '1.0',
        'features': [],
        'last_training_date': 'Unknown',
        'forecast_horizon': 5,
        'data_source': 'Yahoo Finance (yfinance)'
    }
    
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
    
    latest_history = HistoricalPrice.objects.filter(
        stock=stock, 
        date__lt=record.prediction_date
    ).order_by('-date').first()
    
    context = {
        'record': record,
        'stock': stock,
        'latest_observation_date': latest_history.date if latest_history else 'N/A',
        'starting_price': latest_history.close if latest_history else 0.0,
        'generation_date': record.prediction_date.strftime('%Y-%m-%d %H:%M'),
        'model_info': {
            'algorithm': 'XGBoost / RandomForest',
            'version': record.model_version
        }
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
    
    context = {
        'models_list': models_list,
        'active_model': active_model
    }
    return render(request, 'core/model_management.html', context)

@staff_member_required
@transaction.atomic
def activate_model(request, model_id):
    model = get_object_or_404(ModelVersion, id=model_id)
    
    # Check if artifact exists before activating
    if not os.path.exists(model.artifact_path):
        messages.error(request, f"Cannot activate: Artifact not found at {model.artifact_path}")
        return redirect('model_management')
    
    # Deactivate all other models safely
    ModelVersion.objects.filter(is_active=True).update(is_active=False)
    
    # Activate the selected model
    model.is_active = True
    model.save()
    
    messages.success(request, f"Successfully activated model version: {model.version_name}")
    return redirect('model_management')

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
