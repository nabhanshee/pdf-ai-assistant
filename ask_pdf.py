import requests

query = input("Ask something about the PDF: ")
response = requests.post("http://localhost:8080/ask_pdf", json={"query": query})


print("Response:", response.json())
