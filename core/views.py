from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .prediction_engine import generate_5_day_forecast
from .models import PredictionRecord, Stock, HistoricalPrice

def home(request):
    latest_prediction = PredictionRecord.objects.first()
    stock = Stock.objects.filter(ticker='AAPL').first()
    
    historical_data = []
    if stock:
        history = HistoricalPrice.objects.filter(stock=stock).order_by('-date')[:30]
        historical_data = [{'date': h.date.strftime('%Y-%m-%d'), 'close': h.close} for h in reversed(history)]
    
    context = {
        'latest_prediction': latest_prediction,
        'stock': stock,
        'historical_data': historical_data,
        'total_predictions': PredictionRecord.objects.count()
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
    """Displays all past prediction records."""
    records = PredictionRecord.objects.all().order_by('-prediction_date')
    return render(request, 'core/history.html', {'records': records})

def methodology(request):
    """Explains the ML model for academic/defense purposes."""
    return render(request, 'core/methodology.html')

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
