from trl.commands.cli_utils import SFTScriptArguments, TrlParser

from datasets import load_dataset

from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM
)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/nas-alinlp/jialong/models/llama2"
    )
    tokenizer.pad_token = tokenizer.unk_token

    ################
    # Dataset
    ################
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            text = f"{example['input'][i]}\n ### Output: {example['output'][i]}" + "</s>"
            output_texts.append(text)
        return output_texts
    
    response_template = "### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    raw_datasets = load_dataset(path = args.dataset_name)
    train_dataset = raw_datasets[args.dataset_train_split]
    ################
    # Training
    ################
    print(model_config)
    print(get_peft_config(model_config))
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        formatting_func=formatting_prompts_func,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)