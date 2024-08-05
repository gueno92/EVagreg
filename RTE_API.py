import requests
import json
import pandas as pd


class RTE_API:
    """
    Class that handle the data ingestion via RTE API
    """

    def __init__(self) -> None:
        pass

    def generate_token(self):
        """
        Generate the token to access the data

        Description of the process @ https://data.rte-france.com/documents/20182/22648/FR_GuideOauth2_v5.1.pdf/b02d3246-98bc-404c-81c8-dffaad2f1836
        """

        token_url = "https://digital.iservices.rte-france.com/token/oauth/"

        #Authorization is from the created app in the RTE portal
        headers = {
            "Authorization": "Basic OGIwYzljYTQtOGExYi00ZGJmLTlmZmEtYzBmODU0M2NjZDVmOjkxOWY3ODA1LTZkNDQtNDJkYi04NmE3LWVjY2M5NWRkNjFhYg=="
        }
        data = {
            "grant_type": "client_credentials"  # Adjust according to the documentation
        }
        
        # Generate the token via a POST call
        response = requests.post(token_url, headers=headers, data=data)
        
        if response.status_code == 200:
            # If the call worked store the token
            token_response = response.json()
            access_token = token_response.get("access_token")
            self.access_token = access_token
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    # Step 2: Access the API using the Token
    def get_wholesale_data(self):
        """
        Call to get the wholesale data

        Documentation of the API @ https://data.rte-france.com/documents/20182/224298/FR_GU_API_Wholesale_Market_v02.02.pdf
        """
        # Create the basis of the API call
        base_url = "https://digital.iservices.rte-france.com/open_api/wholesale_market/v2"
        endpoint = "/france_power_exchanges"
        url = f"{base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        wholesale_data = requests.get(url, headers=headers)
        
        if wholesale_data.status_code == 200:
            # If API call worked, store result in JSON format
            self.wholesale_data_json = wholesale_data.json()

            #Store it also as a df
            for i, value in enumerate(self.wholesale_data_json['france_power_exchanges'][0]['values']):
                if i == 0:
                    df = pd.DataFrame(value, index=[0])
                else:
                    df_row = pd.DataFrame(value, index=[0])
                    df = pd.concat([df, df_row], axis=0)
            # Reset the index
            df = df.reset_index()
            #Store it in the class
            self.wholesale_data_df = df
        else:
            # If error, raise exception
            raise Exception(f"Error: {wholesale_data.status_code} - {wholesale_data.text}")

    def get_energy_balancing(self,
                            start_date:str,
                            end_date:str):
        """
        Get the energy balancing data via the RTE API.
        Format of start date should be : 2015-06-08T00:00:00

        """
        base_url = "https://digital.iservices.rte-france.com/open_api/balancing_energy/v3/standard_afrr_data?"
        
        url = f"{base_url}start_date={start_date}%2B02:00&end_date={end_date}%2B02:00"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")



    # Step 3: Write Data to a JSON File
    def write_to_json(self, 
                      data, filename="api_data.json"):
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)


