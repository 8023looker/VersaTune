import ujson
import subprocess
from tqdm import tqdm

def check_science_conversation(file_path):
    print("file_path", file_path)
    line_num = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE)  
    line_count = int(line_num.stdout.split()[0])
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
        # for idx, line in enumerate(fin):
        for idx, line in enumerate(tqdm(fin, total=line_count)):
            try:
                content = ujson.loads(line.replace("\n", "").replace("\\/", "/"))
                if content['conversation'][0]["from"] != "system":
                    print(f"{idx}: Not system, error")
            except ValueError as e:
                print(f"JSON parser error: {e}")

def get_example_counts(file_path):
    if file_path.split(".")[-1] == "jsonl":
        line_num = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE)  
        line_count = int(line_num.stdout.split()[0])
        print("line_num", line_num)
    elif file_path.split(".")[-1] == "json":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
            content = ujson.load(fin)
            # print("content", content)
            print(file_path, "len", len(content))
        

check_science_conversation("/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/science/sonnet3.5_science_conversations/sonnet3.5_science_conversations_sharegpt.jsonl") # science
get_example_counts("/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/law/Lawyer-Instruct/alpacmygavel.json") # Lawyer Instruct
get_example_counts("/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json") # medical meadow
get_example_counts("/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/finance/finance-alpaca/Cleaned_date.json") # finance alpaca
get_example_counts("/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/alpaca_rating/alpaca_data.json") # alpaca rating
get_example_counts("/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/code/codeAlpaca/data/code_alpaca_20k.json") # code alpaca