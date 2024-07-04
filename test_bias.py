from deepeval.models.base_model import DeepEvalBaseLLM
import requests
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BiasMetric
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate
from deepeval.metrics import  AnswerRelevancyMetric,ToxicityMetric
from deepeval.dataset import EvaluationDataset
import pandas as pd
import pytest
df = pd.read_csv('output_new_model.csv')
print(df)

dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="output_new_model.csv",
    input_col_name="input_prompt",
    actual_output_col_name="new_output"
)
class CustomModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model = None

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        url="http://44.203.86.100:11434/api/generate"

        response = requests.post(url, json={"model": "mistral",  "prompt": prompt,"stream": False})

        # Check if the request was successful
        # if response.status_code == 200:
        # # Extract the result from the response
        #     result = response.json()["response"]
        # else:
        #     result="Error"
        # print("Answer:", result)
        return response.json()["response"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Custom Model"
code_model = CustomModel()


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    bias_metric = BiasMetric(threshold=0.5, model=code_model)
    assert_test(test_case, [bias_metric])
@deepeval.log_hyperparameters(model="code_model",prompt_template="...")
def hyperparameters():
    return {"temperature": 1, "chunk size": 500}

@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")