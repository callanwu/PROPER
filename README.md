# PROPER

## Requirements

Due to the use of different training frameworks in Stage 1 and other stages, you may need to configure two separate environments. We have listed the requirements for each stage below:

### Stage 1 Requirements

- requirements_stage1.txt

### Other Stages Requirements

- requirement.txt

Sure! Here’s a suggested section for your GitHub README that explains the code modifications you’ve made:

---

## Code Modification Overview

In this project, we have made modifications to the PEFT library to implement the **LoRA-MOE** (Mixture of Experts with LoRA) functionality. Additionally, we have updated the **Transformers** library, specifically in the `generation` and `modeling_llama` modules, to allow for inference based on a `user_id`.

## Key Modifications:

1. **LoRA Implementation**:
   - We have customized the `lora.py` within the PEFT library to enhance its capabilities for handling multiple experts effectively.

2. **Transformers Update**:
   - We also modified the `modeling_llama.py` module and  `generation/utils` module to facilitate the processing of the `user_id` parameter during inference and use the constraint loss, allowing us to tailor the model's responses based on the user.

## Run the PROPER
### Stage1
```bash
conda create -n stage1 python=3.10
conda activate stage1
bash run_loramoe_stage1.sh
```
You need modify these args in `run_loramoe_stage1.sh`:
```
model_name_or_path=
dataset_name=
output_dir=
max_seq_length=
```

Then you can merge the lora then proceed with subsequent stages:
### Stage2
```bash
conda create -n proper python=3.10
conda activate proper
bash run_loramoe_stage2.sh
```
You need modify these args in `run_loramoe_stage2.sh`:
```
pretrained_model=
tokenizer_path=
dataset_dir=
exp_name=
setting=stage2
output_dir=
max_seq_length=
```
### Stage3
```bash
conda activate proper
bash run_loramoe_stage3.1.sh
```

You need modify these args in `run_loramoe_stage3.1.sh`:
```
pretrained_model=
tokenizer_path=
dataset_dir=
peft_config=
output_dir=
exp_name=
setting=stage3.1
```

```bash
conda activate proper
bash run_loramoe_stage3.2.sh
```

You need modify these args in `run_loramoe_stage3.2.sh`:
```
pretrained_model=
tokenizer_path=
dataset_dir=
peft_config=
output_dir=
exp_name=
setting=stage3.2
```

## Inference and Evaluation
See `evaluation` folder