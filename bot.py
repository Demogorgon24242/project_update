import pandas as pd
import os
import requests


def ingest_csv(file_path):
    """
    Ingest a CSV file if it exists and return a DataFrame.
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        print(f"File {file_path} does not exist.")
        return None

def call_model(input_text):
    
    url="http://44.203.86.100:11434/api/generate"

    response = requests.post(url, json={"model": "mistral",  "prompt": input_text,"stream": False})

        # Check if the request was successful
        # if response.status_code == 200:
        # # Extract the result from the response
        #     result = response.json()["response"]
        # else:
        #     result="Error"
        # print("Answer:", result)
    return response.json()["response"]
def generate_outputs(df):
    """
    Generate new outputs using the LLaMA3 model for the input prompts in the DataFrame.
    """
    df['new_output'] = df['input_prompt'].apply(call_model)
    return df

def main():
    # Define the CSV file path
    input_csv = r'old_model_report.csv'  # Replace with your actual CSV file name

    # Ingest the CSV file
    df = ingest_csv(input_csv)
    if df is None:
        return

    # Generate new outputs using the LLaMA3 model
    df = generate_outputs(df)

    # Save the updated DataFrame to a new CSV file
    output_csv = 'output_new_model.csv'
    df.to_csv(output_csv, index=False)
    print(f"New outputs saved to {output_csv}")

if __name__ == "__main__":
    main()
