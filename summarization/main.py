# code based on huggingface examples
import json

from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Seq2SeqTrainingArguments, IntervalStrategy, \
    Seq2SeqTrainer, DataCollatorForSeq2Seq
from summarization.data import load_newsroom


newsroom = load_newsroom(subset=True)
# multi_news = load_dataset("multi_news")
# "google/pegasus-large" - selects most important sentences from text
samples = 32
model_name = "google/pegasus-xsum"
device = "cpu"
src_texts = [str(text) for text in newsroom.data["train"]["text"][:samples]]

tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)


batch = tokenizer(src_texts, truncation=True, padding="longest", return_tensors="pt").to(device)
translated = model.generate(**batch)
tgt_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)


with open("data/example.json", "w") as f:
    json.dump([{
        "text": text,
        "pred_summary": pred
    } for text, pred in zip(src_texts, tgt_texts[:samples])], f, indent=4)


# device = "cuda" if torch.cuda.is_available() else "cpu"
# src_text = [
#     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
# ]
# assert (
#     tgt_text[0]
#     == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
# )

# def preprocess_function(examples):
#     inputs = [doc for doc in examples["text"]]
#     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(examples["summary"], max_length=128, truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
#
# tokenized_newsroom = newsroom.map(preprocess_function, batched=True)
