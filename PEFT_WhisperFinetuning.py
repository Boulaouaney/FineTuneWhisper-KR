from huggingface_hub import interpreter_login

from datasets import load_dataset, DatasetDict

from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from peft import prepare_model_for_int8_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate

import os


def dataset_prep(batch):
    # load audio
    audio = batch["audio"]

    # calculate log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

         # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



if __name__ == "__main__":
    # LOGIN
    interpreter_login()

    # Load Dataset
    Zeroth_Korean = DatasetDict()

    Zeroth_Korean["train"] = load_dataset("Bingsu/zeroth-korean", split="train", use_auth_token=True)
    Zeroth_Korean["test"] = load_dataset("Bingsu/zeroth-korean", split="test", use_auth_token=True)

    print("Before processing:")
    print(Zeroth_Korean)

    # load processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="Korean", task="transcribe")

    # applying data preparation func to all training examples using .map
    Zeroth_Korean = Zeroth_Korean.map(dataset_prep, remove_columns=Zeroth_Korean.column_names["train"], num_proc=1)
    print("After processing:")
    print(Zeroth_Korean)

    # intialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # download metric
    metric = evaluate.load("wer")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Download model in 8bit
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", load_in_8bit=True, device_map="auto")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # preparing model with PEFT
    model = prepare_model_for_int8_training(model, output_imbedding_layer="proj_out")

    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    model = get_peft_model(model, config)
    model.print_trainable_parameters()


    # Define trainnig arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-Large-v2-KR",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=True
    )

    # initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=Zeroth_Korean["train"],
        eval_dataset=Zeroth_Korean["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )
    model.config.use_cache = False  # silence the warnings. Should re-enable for inference!!!

    # save checkpoints
    processor.save_pretrained(training_args.output_dir)


    # start training
    trainer.train()


    # set up args and push to hub
    kwargs = {
        "dataset_tags": "Bingsu/zeroth-korean",
        "dataset": "Zeroth-Korean",
        "language": "ko",
        "model_name": "Whisper Large V2 PEFT KR - BYoussef",
        "finetuned_from": "openai/whisper-large-v2",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }

    trainer.push_to_hub(**kwargs)