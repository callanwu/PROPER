import evaluate
import json
def postprocess_text_classification(preds, labels):
    processed_preds = []
    for pred in preds:
        if "rating: " in str(pred).strip():
            processed_preds.append(str(pred).strip().split(": ")[1])
        else:
            processed_preds.append(str(pred).strip())
    labels = [str(label).strip().split(": ")[1] for label in labels]
    return processed_preds, labels
    
all_labels = ["1","2","3","4","5"]
def create_mapping(x):
    try:
        return all_labels.index(x)
    except:
        # print(x)
        return -1

def compute_metrics(decoded_preds, decoded_labels):
    mae_metric = evaluate.load("./utils/mae.py")
    mse_metric = evaluate.load("./utils/mse.py")
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0
    decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
    decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
    decoded_labels = [create_mapping(x,x) for x in decoded_labels]
    result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared = False)
    result = {"MAE" : result_mae["mae"], "RMSE" : result_rmse["mse"]}
    return result

task = "product_rating"
for check_point in  []:
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
