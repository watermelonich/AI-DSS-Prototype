import requests

def get_healthsage_data(params):
    url = "https://healthsage.ai/api/SomeEndpoint"
    response = requests.get(url, params=params)
    return response.json()




# from pyhealth.datasets import MIMIC3Dataset

# mimic3base = MIMIC3Dataset(
#     # root directory of the dataset
#     root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
#     # raw CSV table name
#     tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
#     # map all NDC codes to CCS codes in these tables
#     code_mapping={"NDC": "CCSCM"},
# )