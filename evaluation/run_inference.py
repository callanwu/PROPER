from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import torch
from tqdm import tqdm
import json
import os
device = "cuda"
tokenizer_path = ""
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,padding_side='left')
tokenizer.pad_token = tokenizer.unk_token
task = ""
task_adic = {"movie_tagging": "tag:","news_categorize": "category:", "news_headline": "headline:", "tweet_paraphrase": "tweet:","citation":"reference: ["}
for check_point in []:
    peft_model_id = ""
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model.to(device)
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.to(device)
    batch_size = 1
    test_data_path = ""
    with open(test_data_path, "r" ,encoding="utf-8") as f:
        datas = []
        for line in f:
            data = json.loads(line)
            datas.append(data)
    predictions = []
    for i in tqdm(range(0,len(datas),batch_size)):
        batch_data = datas[i:i+batch_size]
        prompts = [data["input"] + "\n ### Output: " + task_adic[task] for data in batch_data]
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_ids = model.generate(
            input_ids = model_inputs.input_ids,
            task_types = torch.tensor(data["task_type"]),
            max_new_tokens=10,
            num_beams=1,
            do_sample=False
        )
        responses = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.size(1):], skip_special_tokens=True)
        for data, response in zip(batch_data, responses):
            temp = {}
            temp["gold"] = data["output"]
            temp["prediction"] = response
            print(response)
            predictions.append(temp)
        os.makedirs(""+task, exist_ok=True)
        with open(""+task + "/result"+str(check_point)+".jsonl", "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred)+"\n")