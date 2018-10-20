import requests
from geopy.geocoders import Nominatim
import math

WEATHER_API_KEY = "b9a94d3678fec75c0cdbfe5df1b6a62b"
ON_WATER_KEY = "TpoKkpzH2Zs-KKy--EKS"

def get_wind(location, isFire):
	geolocator = Nominatim(user_agent="")
	location = geolocator.geocode(location)
	lat, lon = location.latitude, location.longitude
	route = "https://api.darksky.net/forecast/372ac58d10033ea53aa580eb6255a0bd/" + str(lat) + ", " + str(lon) + "/"
	r = requests.get(route).json()
	if isFire:
		return [r["currently"]["windBearing"], r["currently"]["windSpeed"], lat, lon]
	return ("No", "Lol No")

def predict_route(wind):
	new_lat_lon = [wind[2] + wind[1]*0.02*math.cos(math.radians(wind[0])), wind[3] + wind[1]*0.02*math.sin(math.radians(wind[0]))]
	r = requests.get("https://api.onwater.io/api/v1/results/" + str(new_lat_lon[0]) + "," + str(new_lat_lon[1]) +  "?access_token=TpoKkpzH2Zs-KKy--EKS")
	if (r.json()["water"]):
		return [0, 0]
	return new_lat_lon
