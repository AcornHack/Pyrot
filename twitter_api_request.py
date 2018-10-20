import requests
from requests_oauthlib import OAuth1

# API_KEY = 
# # API_SECRET = 
# # ACCESS_TOKEN = 
# # ACCESS_TOKEN_SECRET = 

auth = OAuth1(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
def get_location_code(locationName):
	r = requests.get("https://ads-api.twitter.com/4/targeting_criteria/locations?q=" + locationName, auth=auth)
	location = r.json()["data"][0]["targeting_value"]
	return location