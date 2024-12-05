# 正式实验，new version
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ujson
from tqdm import tqdm
import subprocess
import sys

# qwen2.5 72b 需要 4 卡
import os
# tmux a -t qwen_infer
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# tmux a -t qwen_infer_cc_en
os.environ["CUDA_VISIBLE_DEVICES"]="2,4,5,6"

from utils import prompts

    
class QwenInfer:
    def __init__(self, model_path, folder_path):
        self.output_root_folder = folder_path # output_root_folder
        self.llm = LLM(model=model_path, tensor_parallel_size=4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        self.topic_stats = {
            "law": 0,
            "medical": 0,
            "finance": 0,
            "science": 0,
            "code": 0,
            "other": 0
        }
        
    def prompted_infer(self, jsonl_line, query_type):
        content = ujson.loads(jsonl_line.replace("\n", "").replace("\\/", "/"))
        
        raw_text = content["response"]
        prompt = prompts.prompts_dict[query_type]["en"] + "\n\n" + raw_text # use English prompt
            
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # generate outputs
        outputs = self.llm.generate([text], self.sampling_params)
        response = outputs[0].outputs[0].text
        # print(f"Prompt: {prompt!r}, Generated text: {response!r}")
        print(f"Generated text: {response!r}")
        
        # write to file
        topic = self.translate_topic(response, "en")
        if topic != "error":
            self.topic_stats[topic] += 1
            
    def translate_topic(self, topic, language):
        topic = topic.lower() # lowercase
        topic_trans_dict = {
            "zh": {
                "法律": "law",
                "医药": "medical",
                "经济": "finance",
                "理工科学": "science",
                "其他": "other",
                "代码": "code"
            },
           "en": {
                "law": "law",
                "medical && health care": "medical",
                "finance": "finance",
                "science": "science",
                "other": "other",
                "code": "code" 
            }
        }
        if topic not in topic_trans_dict[language]:
            return "error"
        else:
            return topic_trans_dict[language][topic]
    
    def handle_file(self, file_path, query_type):
        base_model_name = file_path.split("/")[-2]
        # get line number
        line_num = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE)  
        line_count = int(line_num.stdout.split()[0])
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
            # for idx, line in enumerate(fin):
            for idx, line in enumerate(tqdm(fin, total=line_count)):
                try:
                    self.prompted_infer(line, query_type)
                except ValueError as e:
                    print(f"JSON parser error: {e}")
            
            # write to file
            output_file_path = self.output_root_folder + f"{base_model_name}_40k.json"
            with open(output_file_path, "w", encoding="utf-8", errors="ignore") as fout: # update
                ujson.dump(self.topic_stats, fout, ensure_ascii = False, indent=4)


if __name__ == "__main__":
    # folder preparation
    output_folder = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/domain_infer/result/"
    os.makedirs(output_folder, exist_ok=True)
    
    # proprietary model (qwen2.5-72B)
    model_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/liyouquan/huggingface_models/Qwen2.5-72B-Instruct/"
    qwen_infer = QwenInfer(model_path, output_folder)
    
    # single file
    # tmux a -t qwen_infer
    # input_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/zks/data/mapneo/cc_zh.0003.jsonl"
    # tmux a -t qwen_infer_cc_en
    # input_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/zks/data/mapneo/cc_en.0005.jsonl"
    
    # file list
    root_folder = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/topic_classification/model_output/"
    base_folder = sys.argv[1] # folder path (model name)
    raw_data_folder = root_folder + base_folder + "/"
    print("raw_data_folder: ", raw_data_folder)
    input_file_list = []
    for dirpath, dirnames, filenames in os.walk(raw_data_folder):
        for i, filename in enumerate(tqdm(filenames, total=len(filenames))):
            file_path = os.path.join(dirpath, filename)
            input_file_list.append(file_path)
    
    for input_file_path in input_file_list:
        query_type = "topic"
        qwen_infer.handle_file(input_file_path, query_type)