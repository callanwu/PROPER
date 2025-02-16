import evaluate
import json
def postprocess_text_classification(preds, labels):
    processed_preds = []
    for pred in preds:
        if "citation: " in str(pred).strip():
            processed_preds.append(str(pred).strip().split(": ")[1])
        else:
            processed_preds.append(str(pred).strip())
    labels = [str(label).strip().split(": ")[1] for label in labels]
    return processed_preds, labels
    
all_labels = ["[1]","[2]"]
def create_mapping(x):
    try:
        return all_labels.index(x)
    except:
        return -1

def compute_metrics(decoded_preds, decoded_labels):
    f1_metric = evaluate.load("./utils/f1.py")
    accuracy_metric = evaluate.load("./utils/accuracy.py")
    decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
    decoded_preds = [create_mapping(x) for x in decoded_preds]
    print(decoded_preds.count(-1))
    decoded_labels = [create_mapping(x) for x in decoded_labels]
    result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, labels=list(range(len(all_labels))), average = "macro")
    result = {"accuracy" : result_acc["accuracy"], "f1" : result_f1["f1"]}
    return result

task = "citation"
for check_point in []:
    with open("result/"+task + "/result"+str(check_point) + ".jsonl") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        predictions = []
        golds = []
        for temp in data:
            predictions.append(temp["prediction"])
            golds.append(temp["gold"])
        result = compute_metrics(predictions,golds)
        print(check_point)
        print(result)
