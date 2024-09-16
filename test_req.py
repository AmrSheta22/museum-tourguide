import requests

# Define the API endpoint
url = "http://127.0.0.1:5000/process"

# Define the data to be sent in the request
data = {
    "artifact_name": "Sirabis",  # Replace with an actual artifact ID
    "question": r"D:\test.wav"  # Replace with your actual question
}

# Send a POST request to the API
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Print the response from the API
    print("Response:", response.json())
else:
    # Print an error message if something went wrong
    print("Failed to get a valid response. Status Code:", response.status_code)
    print("Error Response:", response.text)
