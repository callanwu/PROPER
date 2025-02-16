from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = ""
task = ""
checkpoint = ""
LORA_WEIGHTS = ""
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model, 
    LORA_WEIGHTS, 
    torch_dtype="auto",
    device_map="auto"
)
model = model.merge_and_unload()
model.save_pretrained("")