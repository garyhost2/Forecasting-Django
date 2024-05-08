from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from django.http import JsonResponse
from django.shortcuts import render 

def extract_country_data(df, country_name):
    # Filter the dataframe based on the country name
    country_data = df[df['Country/Region'] == country_name]
    
    # Extract dates and corresponding death counts
    dates = country_data.columns[4:-1]  # Assuming the first 4 columns are non-date related
    deaths = country_data.iloc[:, 4:-1].values.flatten()
    
    return dates, deaths

def train_arima(deaths):
    # Split data into train and test sets (not necessary for this case)
    train_data = deaths
    # Define ARIMA order (p, d, q)
    order = (5, 1, 0)  # Example order, you may need to tune this
    # Fit ARIMA model
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    # Make predictions (using the same training data for simplicity)
    predictions = model_fit.predict(start=len(deaths), end=len(deaths) + 20)
    return predictions

def index(request):
    return render(request, 'main/templates/index.html')

def plot_view(request):
    if 'country' in request.GET:
        country_name = request.GET['country']
        
        # Load your dataframe (df) from your data source
        df = pd.read_csv("data/deaths_global.csv", delimiter=';')
        
        dates, deaths = extract_country_data(df, country_name)
        predictions = train_arima(deaths)
        
        # Convert dates to string format
        dates_str = [str(date) for date in dates]

        # Create a dictionary containing dates, death counts, and predictions
        data = {'dates': dates_str, 'deaths': deaths.tolist(), 'predictions': predictions.tolist()}

        # Convert dictionary to JSON and send as response
        return JsonResponse(data)

    return JsonResponse({'error': 'Country not provided'})
