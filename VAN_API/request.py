import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Country':'India', 'Year':2017, 'Total':89.7})

print(r.json())