import csv
import pandas as pd
import ujson
import numpy as np
import math

def compute_avg_loss(input_loss_file, input_json_file, batch_size=8, gpu_num=4): # csv file
    df = pd.read_csv(input_loss_file)
    dataset_name = input_json_file.split("/")[-1]
    loss_array = list(df[f"../output/proxy_{dataset_name} - train/loss"])
    
    training_count = get_example_counts(input_json_file)
    loss_array = list(chunk_list(loss_array, math.ceil(training_count / (batch_size * gpu_num))))
    print(f"Loss array: {loss_array}")
    print(len(loss_array), len(loss_array[0]))
    
    means = [sum(sublist) / len(sublist) for sublist in loss_array]
    min_mean = min(means)
    print(f"Min Average Loss: {min_mean}")
    
def get_example_counts(file_path):
    if file_path.split(".")[-1] == "jsonl":
        line_num = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE)  
        line_count = int(line_num.stdout.split()[0])
        return line_count
    elif file_path.split(".")[-1] == "json":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
            content = ujson.load(fin)
            return len(content)
        
def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    
if __name__ == "__main__":
    medical_csv_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/proxy_loss/medical_proxy_1b.csv"
    medical_dataset_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json"
    
    finance_csv_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/proxy_loss/finance_proxy_1b.csv"
    finance_dataset_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/finance/finance-alpaca/Cleaned_date.json"
    
    law_csv_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/proxy_loss/law_proxy_1b.csv"
    law_dataset_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/law/Lawyer-Instruct/alpacmygavel.json"
    
    # compute_avg_loss(medical_csv_file_path, medical_dataset_file_path)
    compute_avg_loss(finance_csv_file_path, finance_dataset_file_path)
    # compute_avg_loss(law_csv_file_path, law_dataset_file_path)