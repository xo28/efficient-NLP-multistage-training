import sys
from datasets import load_dataset, ClassLabel, load_metric
from transformers import AutoModelForSequenceClassification, get_scheduler, DataCollatorWithPadding
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
import time
import csv
from tqdm.auto import tqdm
from dataset import to_bin_class, tokenizer, metric_train, metric_test, epoch_time
from dataset import tokenize_fn, tokenize_fn_1, tokenize_fn_2, tokenize_fn_3, tokenize_fn_4
from sklearn.naive_bayes import BernoulliNB

os.environ["WANDB_DISABLED"] = "true"

# hyperparamters
config_batch_size = 8
config_per_sample = True # otherwise per minibatch
config_n_window = config_batch_size * 16 # mov window avg for cal loss threshold, in num of samples
config_layer_mask = [1,1,1,1,1,1]   # per layer. 1=train, 0=freeze
# config_layer_mask = [0,0,0,0,0,1]   # per layer. 1=train, 0=freeze
config_stage2_start = False

# # load pre-trained
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

stage0_steps = int(sys.argv[1])
config_cls_loss = float(sys.argv[2])
config_cls_window_size = int(sys.argv[3])
actual_task = str((sys.argv[4]))
lr = float(sys.argv[5])
gpu_id = str(sys.argv[6])
seed = int(sys.argv[7])

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda:" + gpu_id) if torch.cuda.is_available() else torch.device("cpu")
print(f"This is for task:{actual_task}, steps:{stage0_steps}, 1to2 threshold:{config_cls_loss}, 1to2 windowsize:{config_cls_window_size}")

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())
print('The DistilBERT model has {:} different named parameters.\n'.format(len(params)))

# embeddeing layers
print('==== Embedding Layer ====\n')
for p in params[0:4]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# transformer layers
for layer in range(0, 6):
    # 0th layer: params [5..21]. each layer 16 params, 6 layers
    if layer == 0:
        print('\n==== First Transformer ====\n')
    for p in params[4 + 16 * layer: 20 + 16 * layer]:
        if layer == 0:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        if config_layer_mask[layer] == 0:
            p[1].requires_grad = False  #freeze
# output layer
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

sst = load_dataset('glue', actual_task)

if actual_task in ['cola', 'sst2', 'qqp', "mnli_matched", "qnli", "mrpc","stsb","rte"]:
  sst = sst.remove_columns(['idx'])
else:
  sst = sst.remove_columns(['tokens', 'tree'])
sst = sst.map(to_bin_class)
sst = sst.cast_column('label', ClassLabel(num_classes=2))
if actual_task in ["mrpc","stsb","rte"]:
  sst_tokenized = sst.map(tokenize_fn_1, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['sentence1','sentence2'])
  
elif actual_task in ['sst', 'sst2', 'cola']:
  sst_tokenized = sst.map(tokenize_fn, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['sentence'])

elif actual_task in ['qqp']:
  sst_tokenized = sst.map(tokenize_fn_2, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['question1','question2'])

elif actual_task in ['mnli_matched']:
  sst_tokenized = sst.map(tokenize_fn_3, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['premise','hypothesis'])

elif actual_task in ['qnli']:
  sst_tokenized = sst.map(tokenize_fn_4, batched=True)
  sst_tokenized = sst_tokenized.remove_columns(['question','sentence'])
  
metric = load_metric("glue", actual_task)

if actual_task in ['mnli_matched']:
    trn_set = sst_tokenized['test']
else:
    trn_set = sst_tokenized['train']
num_trn = trn_set.num_rows
tst_set = sst_tokenized['test']
val_set = sst_tokenized['validation']

# data loader, split into train/eval sets
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
   trn_set, shuffle=True, batch_size=config_batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    val_set, batch_size=config_batch_size, collate_fn=data_collator
)

optimizer = AdamW(model.parameters(), lr=lr)
num_epochs = 2 # 1 or 2

num_training_steps = len(train_dataloader)
lr_scheduler = get_scheduler(
"linear",
optimizer=optimizer,
num_warmup_steps=0,
num_training_steps= num_epochs * num_training_steps,
)
print(num_training_steps)

# set stage0 steps from int to percentage
stage0_steps = int(0.01 * stage0_steps * num_training_steps)
model.to(device)

# record skipping status
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
            loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs.logits, batch['labels'])  # per example loss
            loss_history = np.concatenate((loss_history, loss.cpu().detach().numpy()))
            if step_counter < stage0_steps:
                if len(loss_history) > config_n_window:
                    loss_threshold = np.average(loss_history[-config_n_window:])
                loss = outputs.loss
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
                loss = outputs.loss
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
                    loss = torch.nn.CrossEntropyLoss(reduction='none')(output.logits,batch2['labels'])  # per example loss
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

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric_test.add_batch(predictions=predictions, references=batch["labels"])

    val_time = epoch_time(t0, time.time())
    print("  Validation took: {:}m {:}s".format(val_time[0], val_time[1]))
    val_acc = metric_test.compute()['accuracy'] * 100
    print("Validation Accuracy: ", val_acc)
    with open('{}_seed{}.csv'.format(actual_task, seed), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([stage0_steps, config_cls_loss, config_cls_window_size, epoch, transit0_counter, transit1_counter, step_counter, lr, val_acc, \
                         forward_skip_ratio, backward_skip_ratio, total_forward_skip_ratio, total_backward_skip_ratio])


