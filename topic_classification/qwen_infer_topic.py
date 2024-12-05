from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ujson
from tqdm import tqdm
import subprocess

# qwen 72b 需要 4 卡
import os
# tmux a -t qwen_infer
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# tmux a -t qwen_infer_cc_en
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

from utils import prompts

    
class QwenInfer:
    def __init__(self, model_path, foler_path):
        self.output_root_folder = folder_path # output_root_folder
        self.llm = LLM(model=model_path, tensor_parallel_size=4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        
    def prompted_infer(self, jsonl_line, query_type):
        content = ujson.loads(jsonl_line.replace("\n", "").replace("\\/", "/"))
        
        language = content["Language="]
        raw_text = content["text"]
        prompt = prompts.prompts_dict[query_type] # still in dict format
        if language == "zh": # Chinese
            prompt = prompt["zh"] + "\n\n" + raw_text
        else: # English
            prompt = prompt["en"] + "\n\n" + raw_text
            
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
        topic = self.translate_topic(response, language)
        if topic != "error":
            output_file_path = self.output_root_folder + topic + "/" + language + ".jsonl"
            with open(output_file_path, "a", encoding="utf-8", errors="ignore") as fout:
                content["topic"] = topic
                fout.write(ujson.dumps(content, ensure_ascii = False) + "\n")
    
    def translate_topic(self, topic, language):
        topic = topic.lower() # lowercase
        topic_trans_dict = {
            "zh": {
                "法律": "law",
                "医药": "medical",
                "经济": "finance",
                "理工科学": "science",
                "代码": "code",
                "其他": "other"
            },
            "en": {
                "law": "law",
                "medical && health care": "medical",
                "finance": "finance",
                "science": "science",
                "code": "code",
                "other": "other"
            }
        }
        if topic not in topic_trans_dict[language]:
            return "error"
        else:
            return topic_trans_dict[language][topic]
    
    def handle_file(self, file_path, query_type):
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


def make_topic_dirs(folder_path):
    topic_list = {
        "zh": ["法律", "医药", "经济", "理工科学",  "代码", "其他"],
        "en": ["law", "medical", "finance", "science", "code", "other"]
    }
    for topic in topic_list["en"]:
        os.makedirs(os.path.join(folder_path, topic), exist_ok=True) # create the directory
   
 
if __name__ == "__main__":
    # folder preparation
    folder_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/dataset/cpt_dataset/sample_data/common_crawl/"
    make_topic_dirs(folder_path)
    
    model_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/liyouquan/huggingface_models/Qwen2.5-72B-Instruct/"
    qwen_infer = QwenInfer(model_path, folder_path)
    
    # tmux a -t qwen_infer
    # input_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/zks/data/mapneo/cc_zh.0003.jsonl"
    # tmux a -t qwen_infer_cc_en
    input_file_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/zks/data/mapneo/cc_en.0005.jsonl"
    
    query_type = "topic"
    qwen_infer.handle_file(input_file_path, query_type)