import sys
from datasets import load_dataset, load_metric
from transformers import DistilBertModel, get_scheduler, DataCollatorWithPadding
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import csv
from tqdm.auto import tqdm
from dataset import tokenizer, epoch_time
from dataset import tokenize_fn_ag
from sklearn.naive_bayes import BernoulliNB

os.environ["WANDB_DISABLED"] = "true"

# hyperparamters
config_batch_size = 8
config_per_sample = True # otherwise per minibatch
config_n_window = config_batch_size * 16 # mov window avg for cal loss threshold, in num of samples
config_layer_mask = [1,1,1,1,1,1]   # per layer. 1=train, 0=freeze
# config_layer_mask = [0,0,0,0,0,1]   # per layer. 1=train, 0=freeze
config_stage2_start = False
config_cls_window_size = 8

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

## load pre-trained
## multiclass Distillbert
class DistilBERTClass(torch.nn.Module):
    def __init__(self, output_dim):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask, labels):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


stage0_steps = int(sys.argv[1])
config_cls_loss = float(sys.argv[2])
config_cls_window_size = int(sys.argv[3])
actual_task = str((sys.argv[4]))
lr = float(sys.argv[5])
gpu_id = str(sys.argv[6])
device = torch.device("cuda:" + gpu_id) if torch.cuda.is_available() else torch.device("cpu")
print(f"This is for task:{actual_task}, steps:{stage0_steps}, 1to2 threshold:{config_cls_loss}, 1to2 windowsize:{config_cls_window_size}")

dataset = None
model = None
if actual_task in ['ag_news', 'ag']:
    actual_task = "ag_news"
    dataset = load_dataset("ag_news")
    dataset_tokenized = dataset.map(tokenize_fn_ag, batched=True)
    dataset_tokenized.set_format("torch")
    dataset_tokenized = dataset_tokenized.remove_columns(['text'])
    model = DistilBERTClass(4)

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)
metric_test = load_metric("accuracy")
trn_set = dataset_tokenized['train']
num_trn = trn_set.num_rows
val_set = dataset_tokenized['test']

# data loader, split into train/eval sets
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
   trn_set, shuffle=True, batch_size=config_batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    val_set, batch_size=config_batch_size, collate_fn=data_collator
)

optimizer = Adam(model.parameters(), lr=lr)
num_epochs = 2  # 1 or 2

num_training_steps = len(train_dataloader)
lr_scheduler = get_scheduler(
"linear",
optimizer=optimizer,
num_warmup_steps=0,
num_training_steps=num_epochs * num_training_steps,
)
print(num_training_steps)

# set stage0 steps from int to percentage
stage0_steps = int(0.01 * stage0_steps * num_training_steps)
model.to(device)

included_batches = list(range(0, len(train_dataloader)))
total_forward_counter = 0  # how many sample skipped
total_backward_counter = 0
forward_skip_counter = 0
backward_skip_counter = 0
loss_history = np.array([], dtype=np.float32)
loss_history_eff = np.array([], dtype=np.float32)   # effective. actual loss in training
loss_threshold = 0
loss_threshold_history = []

# list of 1D tensors...
staged_batch = {'input_ids':[], 'attention_mask':[], 'labels':[]}

# train/validation per epoch
loss_values = []
loss_threshold_history.append((0, loss_threshold))
bnb = BernoulliNB(binarize=0.0)
classifier_proba = []
step_counter = 0
transit0_counter = -1
transit1_counter = -1

for epoch in range(num_epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0
    classifier_proba = []
    forward_skip_counter = 0
    backward_skip_counter = 0
    if epoch > 0:
        config_stage2_start = True

    progress_bar = tqdm(range(len(train_dataloader) * config_batch_size))
    for step, batch in enumerate(train_dataloader):
        print("Loss threshold: "+str(loss_threshold))
        batch = {k: v.to(device) for k, v in batch.items()}
        batch1 = {}
        batch2 = {}
        # Naive Bayes Classification
        bow = np.zeros((batch['input_ids'].shape[0], tokenizer.vocab_size))
        for i in range(batch['input_ids'].shape[0]):
            for index, j in enumerate(batch['input_ids'][i]):
                if batch['attention_mask'][i][index] == 1:
                    bow[i][j] += 1
        features = bow
        if not config_stage2_start:
            model.train()
            outputs = model(**batch)
            # per sample losses
            loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs, batch['labels'])  # per example loss
            loss_history = np.concatenate((loss_history, loss.cpu().detach().numpy()))
            if step_counter < stage0_steps:
                if len(loss_history) > config_n_window:
                    loss_threshold = np.average(loss_history[-config_n_window:])
                loss = loss_fn(outputs, batch['labels'])
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
            else:
                if transit0_counter < 0:
                    transit0_counter = step_counter
                target = np.array(loss.detach().cpu().numpy() > loss_threshold).astype(np.int32)
                bnb.partial_fit(features,y=target, classes=np.array([0,1]))
                pred = bnb.predict(features)
                proba = -np.average(bnb.predict_log_proba(features)[np.arange(batch['input_ids'].shape[0]),target])
                classifier_proba.append(proba)
                # transition of stage 1 to stage 2
                if len(classifier_proba) >= config_cls_window_size:
                    print(('Average Classifier Loss in Window',
                           sum(classifier_proba[-config_cls_window_size:]) / config_cls_window_size))
                    if sum(classifier_proba[-config_cls_window_size:]) / config_cls_window_size <= config_cls_loss:
                        config_stage2_start = True
                ###############Stage 1################
                for idx, l in enumerate(loss):
                    if l >= loss_threshold:
                        print('stage 11111')
                        for k in ['input_ids', 'attention_mask', 'labels']:
                            # staged_batch[k] = torch.cat(staged_batch[k], batch[k][idx])
                            staged_batch[k].append(batch[k][idx])
                    else:
                        backward_skip_counter += 1

                n_batches = len(staged_batch['input_ids'])
                if n_batches < config_batch_size:
                    continue

                for k in ['input_ids', 'attention_mask', 'labels']:
                    batch1[k] = torch.stack(staged_batch[k][0:config_batch_size]).to(device)  # already on device??
                    staged_batch[k] = staged_batch[k][config_batch_size:]

                model.train()
                outputs = model(**batch1)
                loss = loss = loss_fn(outputs, batch1['labels'])
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

        ###############Stage 2################
        else:
            if transit1_counter < 0:
                transit1_counter = step_counter
            pred = bnb.predict(features)
            # backprop based on loss threshold
            print('stage 22222')
            for idx, l in enumerate(pred):
                if l == 1:
                    model.eval()
                    for k in ['input_ids', 'attention_mask', 'labels']:
                        batch2[k] = batch[k][idx].unsqueeze(0)  # already on device??
                    output = model(**batch2)
                    loss = torch.nn.CrossEntropyLoss(reduction='none')(output,batch2['labels'])  # per example loss
                    target = np.array(loss.detach().cpu().numpy() > loss_threshold).astype(np.int32)
                    bnb.partial_fit(features[idx][np.newaxis,:], y=target, classes=np.array([0, 1]))
                    if loss > loss_threshold:
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        total_loss += loss.item()
                    else:
                        backward_skip_counter += 1
                else:
                    forward_skip_counter += 1
                    backward_skip_counter += 1

        step_counter += 1
        progress_bar.update(config_batch_size)
    if epoch == 0 and not config_stage2_start:
        exit()
    avg_train_loss = total_loss / (len(train_dataloader) - backward_skip_counter / config_batch_size)  # xzl
    loss_values.append(avg_train_loss)

    print("  Average training loss: {:}".format(avg_train_loss))
    train_time = epoch_time(t0, time.time())
    print("  Training epcoh took: {:}m {:}s".format(train_time[0], train_time[1]))
    total_forward_counter += forward_skip_counter
    total_backward_counter += backward_skip_counter
    total_forward_skip_ratio = 100 * total_forward_counter / config_batch_size / len(train_dataloader) / num_epochs
    total_backward_skip_ratio = 100 * total_backward_counter / config_batch_size / len(train_dataloader) / num_epochs
    forward_skip_ratio = 100 * forward_skip_counter / config_batch_size / len(train_dataloader)
    backward_skip_ratio = 100 * backward_skip_counter / config_batch_size / len(train_dataloader)
    print(f" forward skip ratio {forward_skip_ratio:.2f}%, backward skip ratio {backward_skip_ratio:.2f}%",
          "loss_threshold", loss_threshold)
    print(f" Total forward skip ratio {total_forward_skip_ratio:.2f}%, total backward skip ratio {total_backward_skip_ratio:.2f}%",
          "loss_threshold", loss_threshold)

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        # logits = loss_fn(outputs, batch['labels'])
        predictions = torch.argmax(outputs, dim=-1)
        metric_test.add_batch(predictions=predictions, references=batch["labels"])

    val_time = epoch_time(t0, time.time())
    print("  Validation took: {:}m {:}s".format(val_time[0], val_time[1]))
    val_acc = metric_test.compute()['accuracy'] * 100
    print("Validation Accuracy: ", val_acc)
    with open('{}_seed.csv'.format(actual_task, seed), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([stage0_steps, config_cls_loss, config_cls_window_size, epoch, transit0_counter, transit1_counter, step_counter, lr, val_acc, \
                         forward_skip_ratio, backward_skip_ratio, total_forward_skip_ratio, total_backward_skip_ratio])