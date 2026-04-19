# test_api.py (Ek alag file banakar run karein)
import requests

data = {"product_name": "Laptop"}
response = requests.post("http://127.0.0.1:5000/recommend", json=data)
print(response.json())