from django.shortcuts import render
import requests
from .models import City
from .forms import CityForm

def index(request):
    apikey = 'PUT_YOUR_APIKEY'
    base_url = 'https://api.openweathermap.org/data/2.5/weather?'
    city = 'Budapest'
    params = '&units=metric'
    url = base_url + 'q={}' + params + '&appid=' + apikey
    cities = City.objects.all() #return all the cities in the database

    if request.method == 'POST': # only true if form is submitted
        form = CityForm(request.POST) # add actual request data to form for processing
        form.save() # will validate and save if validate

    form = CityForm() # create the form to get city name info

    weather_data = [] # weather data of all cities

    for city in cities:

        city_weather = requests.get(url.format(city)).json() # city weather API response in json format

        weather = { # build a dictionary from the API response, update city, temperature, description and icon
            'city' : city,
            'temperature' : city_weather['main']['temp'],
            'description' : city_weather['weather'][0]['description'],
            'icon' : city_weather['weather'][0]['icon']
        }

        weather_data.append(weather) # add the data for the current city into the list

    context = {'weather_data' : weather_data, 'form' : form} # this will update the template with current information

    return render(request, 'weather/index.html', context) # returns the index.html template
