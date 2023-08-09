from datetime import datetime
import requests


def send_data_to_database(input_name, input_timestamp):
    print("running send_data")
    

    # register data into JSON
    data = {
        "name": input_name,
        "timestamp": input_timestamp
    }
    # POST METHOD
    url = 'http://localhost:3000/model'  # Replace with the actual URL of your database
    response = requests.post(url, json=data)
    print(response.text)

    #if response.status_code == 200:
    #    print('Data sent successfully!')
    #else:
    #    print('Error sending data to nodejs.')