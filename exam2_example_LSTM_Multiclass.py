# package
import torch
from datasets import load_dataset, load_metric, DatasetDict, Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_scheduler, AdamW, DataCollatorWithPadding
from transformers.modeling_outputs import TokenClassifierOutput
from collections import Counter

################################### Gloal Variable #########################################
NUM_EPOCH = 3
NUM_LABEL = 4
BATCH_SIZE = 8
checkpoint = "bert-base-cased"
################################### Data preporcessing #########################################
# %%
# load dataset
dataset_train = load_dataset("ag_news", split='train')
dataset_train = Dataset.from_dict(dataset_train[:1000])
dataset_train_val = dataset_train.train_test_split(test_size=0.2, shuffle=True)
dataset_test = load_dataset("ag_news", split='test')

data = DatasetDict({
    'train': dataset_train_val['train'],
    'val': dataset_train_val['test'],
    'test': dataset_test})

# choice tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len = 512


# choice the column we want to tokenize
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# After using tokenizer, there will appear columns 'input_ids', 'token_type_ids', 'attention_mask'
tokenized_dataset = data.map(preprocess_function, batched=True)
# Data collator, use to padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# train and val and test
tokenized_dataset = tokenized_dataset.remove_columns(['text', 'token_type_ids'])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")
print(tokenized_dataset["train"].column_names)
# create data loader
train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["val"], batch_size=BATCH_SIZE, collate_fn=data_collator
)
test_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=BATCH_SIZE, collate_fn=data_collator
)
######################################## Model Definition ########################################
"""
    from https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd
"""
class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels
        # Load Model with given checkpoint and extract its body
        self.bert = AutoModel.from_pretrained(checkpoint,
                                              config=AutoConfig.from_pretrained(checkpoint,
                                                                                output_attentions=True,
                                                                                output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256 * 2, num_labels)  # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions'])
        # outputs[0] = [batch , ? , 768]
        lstm_output, (h, c) = self.lstm(outputs[0])  ## extract the 1st token's embeddings
        # torch.Size([8, ?, 512]) 512 because bidirectional=True
        # lstm_output[:, -1, :256].shape = torch.Size([8, 256])
        # lstm_output[:, 0, 256:].shape = torch.Size([8, 256])
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        # hidden.shape = torch.Size([8, 512])
        logits = self.linear(hidden.view(-1, 256 * 2))  ### assuming that you are only using the output of the last LSTM cell to perform classification
        # logits shape [batch * num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)


######################################## Model Initialization ########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel(checkpoint=checkpoint, num_labels=NUM_LABEL).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = NUM_EPOCH
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
######################################## Train Data ########################################
# metric_f1 = load_metric("f1")
metric_acc = load_metric("accuracy")

progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # metric_f1.add_batch(predictions=predictions, references=batch["labels"])
        metric_acc.add_batch(predictions=predictions, references=batch["labels"])

        progress_bar_eval.update(1)

    # print(metric_f1.compute())
    print(metric_acc.compute())
######################################## Test Data ########################################
model.eval()
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    # metric_f1.add_batch(predictions=predictions, references=batch["labels"])
    metric_acc.add_batch(predictions=predictions, references=batch["labels"])

# print(metric_f1.compute())
print(metric_acc.compute())
