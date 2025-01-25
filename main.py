import warnings
import os
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import random
import os
import argparse
import datetime
from scheduler import LinearDecayLR

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from rouge import Rouge
from torch.optim import Adam

from data import MultiTaskDataset
from model import *
from logger import Logger


WEIGHTING_SETTING = {
    "0": "static",
    "1": "relative",
    "2": "grad"
}

def evaluate_model(model, dataloader, tokenizer, args, logger, max_summary_length=40):
    model.eval()
    rouge = Rouge()

    task2_rouge_1_scores = []
    task2_rouge_2_scores = []
    task2_rouge_l_scores = []

    abstractive_rouge_1_scores = []
    abstractive_rouge_2_scores = []
    abstractive_rouge_l_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            task2_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            abstractive_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

            for i in range(len(batch['task'])):
                task = batch['task'][i]
                if task == 'task2':
                    task2_inputs["input_ids"].append(batch["input_ids"][i].to(args.device))
                    task2_inputs["attention_mask"].append(batch["attention_mask"][i].to(args.device))
                    task2_inputs["labels"].append(batch["labels"][i].to(args.device))
                elif task == 'abstractive':
                    abstractive_inputs["input_ids"].append(batch["input_ids"][i].to(args.device))
                    abstractive_inputs["attention_mask"].append(batch["attention_mask"][i].to(args.device))
                    abstractive_inputs["labels"].append(batch["labels"][i].to(args.device))

            if task2_inputs["input_ids"]:
                task2_inputs = {k: torch.stack(v) for k, v in task2_inputs.items()}
                task2_outputs = model.generate(input_ids=task2_inputs["input_ids"], attention_mask=task2_inputs["attention_mask"], max_length=max_summary_length, task='task2')
                generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in task2_outputs]

                target_texts = []
                for t in task2_inputs["labels"]:
                    t = t[t != -100]  # Remove -100 tokens
                    target_texts.append(tokenizer.decode(t, skip_special_tokens=True))

                for g_text, t_text in zip(generated_texts, target_texts):
                    scores = rouge.get_scores(g_text, t_text)[0]
                    task2_rouge_1_scores.append(scores['rouge-1']['f'])
                    task2_rouge_2_scores.append(scores['rouge-2']['f'])
                    task2_rouge_l_scores.append(scores['rouge-l']['f'])

            if abstractive_inputs["input_ids"]:
                abstractive_inputs = {k: torch.stack(v) for k, v in abstractive_inputs.items()}
                abstractive_outputs = model.generate(input_ids=abstractive_inputs["input_ids"], attention_mask=abstractive_inputs["attention_mask"], max_length=max_summary_length, task='abstractive')
                generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in abstractive_outputs]

                target_texts = []
                for t in abstractive_inputs["labels"]:
                    t = t[t != -100]  # Remove -100 tokens
                    target_texts.append(tokenizer.decode(t, skip_special_tokens=True))

                for g_text, t_text in zip(generated_texts, target_texts):
                    scores = rouge.get_scores(g_text, t_text)[0]
                    abstractive_rouge_1_scores.append(scores['rouge-1']['f'])
                    abstractive_rouge_2_scores.append(scores['rouge-2']['f'])
                    abstractive_rouge_l_scores.append(scores['rouge-l']['f'])

    # Calculate average ROUGE scores
    avg_task2_rouge_1 = sum(task2_rouge_1_scores) / len(task2_rouge_1_scores) if task2_rouge_1_scores else 0
    avg_task2_rouge_2 = sum(task2_rouge_2_scores) / len(task2_rouge_2_scores) if task2_rouge_2_scores else 0
    avg_task2_rouge_l = sum(task2_rouge_l_scores) / len(task2_rouge_l_scores) if task2_rouge_l_scores else 0

    avg_abstractive_rouge_1 = sum(abstractive_rouge_1_scores) / len(abstractive_rouge_1_scores) if abstractive_rouge_1_scores else 0
    avg_abstractive_rouge_2 = sum(abstractive_rouge_2_scores) / len(abstractive_rouge_2_scores) if abstractive_rouge_2_scores else 0
    avg_abstractive_rouge_l = sum(abstractive_rouge_l_scores) / len(abstractive_rouge_l_scores) if abstractive_rouge_l_scores else 0

    if args.single_task == 0:
        logger("Task2 ROUGE Scores:")
        logger(f"  - ROUGE-1: {avg_task2_rouge_1}")
        logger(f"  - ROUGE-2: {avg_task2_rouge_2}")
        logger(f"  - ROUGE-L: {avg_task2_rouge_l}")

    logger("Abstractive ROUGE Scores:")
    logger(f"  - ROUGE-1: {avg_abstractive_rouge_1}")
    logger(f"  - ROUGE-2: {avg_abstractive_rouge_2}")
    logger(f"  - ROUGE-L: {avg_abstractive_rouge_l}")

    return {
        "task2": {
            "rouge-1": avg_task2_rouge_1,
            "rouge-2": avg_task2_rouge_2,
            "rouge-l": avg_task2_rouge_l
        },
        "abstractive": {
            "rouge-1": avg_abstractive_rouge_1,
            "rouge-2": avg_abstractive_rouge_2,
            "rouge-l": avg_abstractive_rouge_l
        }
    }

def train(
        model, 
        train_dataloader,
        valid_dataloader,
        tokenizer,
        optimizer_main,

        task2_loss_weight, 
        summarization_loss_weight, 

        args,
        logger,

        num_epochs=20, 
        max_grad_norm=1.0, 
        eval_accumulation_steps=None,

        omega=None,
        omega_optim=None,

        weights=None,
        optimizer_weights=None
    ):

    best_abstractive_rougel = 0
    stagnant_epochs_abstractive = 0
    avg_grad_norms = torch.zeros(2).to(args.device)
    lr_scheduler=LinearDecayLR(optimizer_main, num_epochs, 15)

    for epoch in range(num_epochs):
        model.train()
        total_loss_task2 = 0.0
        total_loss_abstractive = 0.0

        for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            optimizer_main.zero_grad()
            if omega_optim is not None: omega_optim.zero_grad()

            # task2_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "decoder_attention_mask": []}
            # abstractive_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "decoder_attention_mask": []}
            task2_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            abstractive_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

            for i in range(len(batch['task'])):
                task = batch['task'][i]
                if task == 'task2':
                    task2_inputs["input_ids"].append(batch["input_ids"][i])
                    task2_inputs["attention_mask"].append(batch["attention_mask"][i])
                    task2_inputs["labels"].append(batch["labels"][i])
                    # task2_inputs["decoder_attention_mask"].append(batch["decoder_attention_mask"][i])
                elif task == 'abstractive':
                    abstractive_inputs["input_ids"].append(batch["input_ids"][i])
                    abstractive_inputs["attention_mask"].append(batch["attention_mask"][i])
                    abstractive_inputs["labels"].append(batch["labels"][i])
                    # abstractive_inputs["decoder_attention_mask"].append(batch["decoder_attention_mask"][i])
                
            task2_loss = 0
            if task2_inputs["input_ids"]:
                task2_inputs = {k: torch.stack(v).to(args.device) for k, v in task2_inputs.items()}
                task2_outputs = model(
                    input_ids=task2_inputs["input_ids"],
                    attention_mask=task2_inputs["attention_mask"],
                    labels=task2_inputs["labels"],
                    # decoder_attention_mask=task2_inputs["decoder_attention_mask"],
                    task='task2'
                )
                task2_loss = task2_outputs.loss
                total_loss_task2 += task2_loss.item()

            abstractive_loss = 0
            if abstractive_inputs["input_ids"]:
                abstractive_inputs = {k: torch.stack(v).to(args.device) for k, v in abstractive_inputs.items()}
                abstractive_outputs = model(
                    input_ids=abstractive_inputs["input_ids"],
                    attention_mask=abstractive_inputs["attention_mask"],
                    labels=abstractive_inputs["labels"],
                    # decoder_attention_mask=abstractive_inputs["decoder_attention_mask"],
                    task='abstractive'
                )
                abstractive_loss = abstractive_outputs.loss
                total_loss_abstractive += abstractive_loss.item()

            if omega_optim is not None:
                total_loss = (omega * abstractive_loss) + ((omega / 2) * task2_loss)
            elif optimizer_weights is not None:
                if isinstance(task2_loss, int):
                    task2_loss = torch.tensor(0.0, device=args.device, requires_grad=True)

                task2_loss.backward(retain_graph=True)
                grad_norms_task2 = torch.tensor([torch.norm(param.grad.detach(), 2) for param in model.parameters() if param.grad is not None], device=args.device).mean()
            
                abstractive_loss.backward(retain_graph=True)
                grad_norms_abstractive = torch.tensor([torch.norm(param.grad.detach(), 2) for param in model.parameters() if param.grad is not None], device=args.device).mean()

                avg_grad_norms[0] = 0.9 * avg_grad_norms[0] + 0.1 * grad_norms_task2
                avg_grad_norms[1] = 0.9 * avg_grad_norms[1] + 0.1 * grad_norms_abstractive

                target_norm = avg_grad_norms.mean()
                scaling_factor_task2 = (grad_norms_task2 / target_norm).mean().item()
                scaling_factor_abstractive = (grad_norms_abstractive / target_norm).mean().item()
                adjusted_loss_task2 = scaling_factor_task2 * weights[0] * task2_loss
                adjusted_loss_abstractive = scaling_factor_abstractive * weights[1] * abstractive_loss

                total_loss = adjusted_loss_task2 + adjusted_loss_abstractive
            else:    
                total_loss = (task2_loss_weight * task2_loss) + (summarization_loss_weight * abstractive_loss)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer_main.step()
            optimizer_main.zero_grad()

            if omega_optim is not None:
                omega_optim.step()
                omega_optim.zero_grad()

        rouge_scores = evaluate_model(model, valid_dataloader, tokenizer, args, logger)
        if rouge_scores is None:
            logger("Skipping evaluation due to errors.")
            continue

        lr_scheduler.step()

        current_abstractive_rougel = rouge_scores["abstractive"]["rouge-l"]
        if current_abstractive_rougel > best_abstractive_rougel:
            best_abstractive_rougel = current_abstractive_rougel
            stagnant_epochs_abstractive = 0
            torch.save(model.state_dict(), f"./models/model.pth")
            print("New Best Score ; Model Saved")
        else:
            stagnant_epochs_abstractive += 1

        if args.single_task == 1:
            logger(f"Epoch [{epoch+1}/{num_epochs}] - Abstractive Loss: {total_loss_abstractive/len(train_dataloader):.4f} - Best Abstractive ROUGE-L: {best_abstractive_rougel:.4f} - Patience: {5-stagnant_epochs_abstractive} - LR:  {lr_scheduler.get_lr()[0]}")
        else:
            logger(f"Epoch [{epoch+1}/{num_epochs}] - Task2 Loss: {total_loss_task2/len(train_dataloader):.4f} - Abstractive Loss: {total_loss_abstractive/len(train_dataloader):.4f}  - Best Abstractive ROUGE-L: {best_abstractive_rougel:.4f} - Patience: {5-stagnant_epochs_abstractive}")

        logger("=================================\n")

        if stagnant_epochs_abstractive >= 5:
            logger("Early stopping as abstractive task has not improved for 3 consecutive epochs.")
            return model



def main(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    #FOR MAC --------------------------------------------
    if args.device == "mps":
        torch.mps.manual_seed(seed)
        torch.backends.mps.deterministic=True
        torch.backends.mps.benchmark = False

    #FOR WINDOWS AND LINUX ------------------------------
    if args.device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True #replace mps with cudnn here
        torch.backends.cudnn.benchmark = False #replace mps with cudnn here
    
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if WEIGHTING_SETTING[args.weighting_setting] == "static" and args.single_task == 1:
        output_file = f"./outputs/{args.model}_aw{args.abstractive_weight}_ew{args.extractive_weight}_dsl{args.decoder_split_level}_bs{args.batch_size}_{date}.txt"
    elif args.single_task == 0:
        output_file = f"./outputs/{args.model}_ws{WEIGHTING_SETTING[args.weighting_setting]}_dsl{args.decoder_split_level}_bs{args.batch_size}_{date}.txt"
    else:
        output_file = f"./outputs/{args.model}_st{args.single_task}_bs{args.batch_size}_{date}.txt"

    if os.path.exists(output_file):
        os.remove(output_file)

    model_name = "UBC-NLP/AraT5-base"
    if args.model_version == "2":
        model_name = "UBC-NLP/AraT5v2-base-1024"

    logger = Logger(output_file)
    logger("CONFIGS:")
    logger(f"Model Name: {model_name}")
    logger(f"Model: {args.model}")
    logger(f"Weighting Setting: {WEIGHTING_SETTING[args.weighting_setting]}")
    logger(f"Single Task: {args.single_task==1}")

    if WEIGHTING_SETTING[args.weighting_setting] == "static" and args.single_task == 0:
        logger(f"Abstractive Weight: {args.abstractive_weight}")
        logger(f"Extractive Weight: {args.extractive_weight}")
        
    logger(f"Decoder Split Level: {args.decoder_split_level}")
    logger(f"Batch Size: {args.batch_size}")
    logger(f"Epochs: {args.epochs}")
    logger(f"Device: {args.device}")
    logger("===============================================\n")

    # MODEL LOADING -------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AraT5_PMTL(T5ForConditionalGeneration.from_pretrained(model_name, resume_download=True)).to(args.device)

    optimizer = Adam(model.parameters(), lr = 5e-5, weight_decay=1e-5)
    # MODEL LOADING -------------------------------------

    # DATA LOADING --------------------------------------
    train_data_abs = pd.read_csv('./data/train1_processed.csv')
    valid_data_abs = pd.read_csv('./data/valid_processed.csv')
    test_data_abs= pd.read_csv('./data/test_processed.csv')
    ood_data = pd.read_csv('./data/ood_processed.csv')
    task2_data = pd.read_csv('./data/paraphrasing_data.csv', delimiter=';')
    # task2_data = pd.read_csv('./data/extractive_data.csv')

    train_data_task2, temp_data_task2 = train_test_split(task2_data, test_size=0.1, random_state=42)
    valid_data_task2, test_data_task2 = train_test_split(temp_data_task2, test_size=0.5, random_state=42)

    # drop nans
    train_data_abs = train_data_abs.dropna()
    valid_data_abs = valid_data_abs.dropna()
    test_data_abs = test_data_abs.dropna()
    ood_data = ood_data.dropna()
    train_data_task2 = train_data_task2.dropna()
    valid_data_task2 = valid_data_task2.dropna()
    test_data_task2 = test_data_task2.dropna()

    train_dataset = MultiTaskDataset(tokenizer, task2_data=train_data_task2 if args.single_task == 0 else None, summarization_data=train_data_abs)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    validation_dataset = MultiTaskDataset(tokenizer, task2_data=None, summarization_data=valid_data_abs)
    valid_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = MultiTaskDataset(tokenizer, task2_data= None, summarization_data=test_data_abs)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ood_dataset = MultiTaskDataset(tokenizer, task2_data=None, summarization_data=ood_data)
    ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger("Data loaded successfully")
    logger(f"EXTRACTIVE TRAIN DATA:{train_data_abs.shape}")
    logger(f"EXTRACTIVE VALID DATA:{valid_data_abs.shape}")
    logger(f"EXTRACTIVE TEST DATA:{test_data_abs.shape}")
    logger("\n")
    logger(f"ABSTRACTIVE TRAIN DATA:{train_data_task2.shape}")
    logger(f"ABSTRACTIVE VALID DATA:{valid_data_task2.shape}")
    logger(f"ABSTRACTIVE TEST DATA:{test_data_task2.shape}")
    logger("\n")
    logger(f"OOD DATA:{ood_data.shape}")
    # DATA LOADING --------------------------------------

    logger("\nTraining=====================================\n")

    omega, omega_optim = None, None
    weights, optimizer_weights = None, None
    if WEIGHTING_SETTING[args.weighting_setting] == "relative":
        omega = nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)
        omega_optim = torch.optim.AdamW([omega], lr=5e-6)
    elif WEIGHTING_SETTING[args.weighting_setting] == "grad":
        weights = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        optimizer_weights = AdamW([weights], lr=5e-6)

    train(
        model, 
        train_dataloader, 
        test_dataloader, 
        tokenizer, 
        optimizer, 
        args.abstractive_weight, 
        args.extractive_weight, 
        args,
        logger,
        num_epochs=args.epochs,
        omega=omega,
        omega_optim=omega_optim,
        weights=weights,
        optimizer_weights=optimizer_weights
    )

    logger("\n========================================\n")
    logger("Evaluating on Test Data")
    evaluate_model(model, test_dataloader, tokenizer, args, logger)

    logger("\n========================================\n")
    logger("Evaluating on OOD Data")
    evaluate_model(model, ood_dataloader, tokenizer, args, logger)

    

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',dest='model', default='parallel')
    parser.add_argument('--model_version',dest='model_version', default='1')
    parser.add_argument('--single_task', dest='single_task', default='0')
    parser.add_argument('--weighting_setting', dest='weighting_setting', default='0')
    parser.add_argument('--abstractive_weight', dest='abstractive_weight', default='1.0')
    parser.add_argument('--extractive_weight', dest='extractive_weight', default='1.0')
    parser.add_argument('--decoder_split_level', dest='decoder_split_level', default='1')
    parser.add_argument('--batch_size', dest='batch_size', default='8')
    parser.add_argument('--epochs', dest='epochs', default='35')
    parser.add_argument('--device', dest='device', default='cuda')
    args=parser.parse_args()

    args.batch_size = int(args.batch_size)
    args.epochs = int(args.epochs)
    args.single_task = int(args.single_task)
    args.decoder_split_level = int(args.decoder_split_level)
    args.abstractive_weight = float(args.abstractive_weight)
    args.extractive_weight = float(args.extractive_weight)

    if not torch.cuda.is_available():
        args.device='mps'

    if not os.path.exists("./models"):
        os.makedirs("./models")

    main(args)