import os
import json
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.server_api import ServerApi
def extract_alert_data(log_file_path):
    with open(log_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    alerts = data['alerts']
    extracted_data = []
 
    for alert in alerts:
        timestamp = alert.get('startsAt') or "N/A"  # If 'startsAt' is None or empty, set to "N/A"
        labels = alert.get('labels', {})
        instance = labels.get('instance') or "N/A"  # If 'instance' is None or empty, set to "N/A"
        alertname = labels.get('alertname') or "N/A"  # If 'alertname' is None or empty, set to "N/A"
        # value = data['alerts']['values'] or "N/A"  # If 'valueString' is None or empty, set to "N/A"
        extracted_data.append({
            'Timestamp': timestamp,
            'Instance': instance,
            'Alertname': alertname
        })
    
    return extracted_data


    
def extract_from_mongo(log_dataframe):
    # Remove rows with any 'N/A' values
    input_df = log_dataframe.replace('N/A', pd.NA).dropna()

    uri = "mongodb+srv://aditya_242:lpAZMJa7cVCgzuKw@cluster0.bfasavc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client['Project']
    collection = db['chat']
    # Prepare the output data
    output_data = []

    # Loop through each row in the input dataframe
    for index, row in input_df.iterrows():
        timestamp = row['Timestamp']
        instance = row['Instance']
        alertname = row['Alertname']

        # Extract date part only (year-month-day) from the timestamp
        date_str = timestamp.split('T')[0]
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        # Query the MongoDB collection using the date part only
        query_result = collection.find_one({
            'timestamp': {
                '$gte': datetime(date_obj.year, date_obj.month, date_obj.day),
                '$lt': datetime(date_obj.year, date_obj.month, date_obj.day) + pd.Timedelta(days=1)
            }
        })

        if query_result:
            output_data.append({
                'id': str(query_result['_id']),
                'input_prompt': query_result['input'],
                'output_prompt': query_result['output'],
                'metric_score': query_result['json_payload']['score'],
                'metric_reason': query_result['json_payload']['reasons'],
                'flagged_metric': alertname,
                'flagged_instance': instance
            })

    # Create a DataFrame from the output data
    output_df = pd.DataFrame(output_data)
    return output_df
    
    
def main():
    folder_path = r"up_alert_logs"  # Use raw string to avoid unicode errors
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            all_data.extend(extract_alert_data(file_path))
    
    try:
        df = pd.DataFrame(all_data)
        result= extract_from_mongo(df)
        result.to_csv("old_model_report.csv",index=False)
    
        print('DataFrame created successfully')
    except:
        print("There was error while creating old_model_report")

if __name__ == "__main__":
    main()
