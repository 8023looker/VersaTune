import ujson

def compute_ratio(input_file_path):
    with open(input_file_path, "r", encoding="utf-8", errors="ignore") as fin:
        content = ujson.load(fin) # dict format
        print(content)
        total_count = 0
        for key in content:
            total_count += content[key]
        for key in content:
            print(f"{key}: {content[key]} / {total_count} = {content[key] / total_count:.2f}")
        
compute_ratio("/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/SFT_series/domain_infer/result/baichuanSEED_40k_code.json")