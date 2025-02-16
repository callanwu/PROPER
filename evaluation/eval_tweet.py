import evaluate
import json
rouge_metric = evaluate.load('./utils/rouge.py')
def postprocess_text_generation(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
def compute_metrics(decoded_preds, decoded_labels):
    decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
    result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
    return result

task = "tweet_paraphrase"
for check_point in range():
    with open("result/"+task + "/result"+str(check_point)+".jsonl") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        predictions = []
        golds = []
        for temp in data:
            predictions.append(temp["prediction"])
            golds.append(temp["gold"].strip().split(": ")[1])
        # print(golds)
        result = compute_metrics(predictions,golds)
        print(result)
        print(check_point)
