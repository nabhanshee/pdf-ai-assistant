import requests

file_path = "pdf/mementopython3-english.pdf"

response = requests.post(
    "http://localhost:8080/pdf",
    files={"file": open(file_path, "rb")}
)

print(response.json())
