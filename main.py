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

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from rouge import Rouge

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

    qa_rouge_1_scores = []
    qa_rouge_2_scores = []
    qa_rouge_l_scores = []

    abstractive_rouge_1_scores = []
    abstractive_rouge_2_scores = []
    abstractive_rouge_l_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            qa_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            abstractive_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

            for i in range(len(batch['task'])):
                task = batch['task'][i]
                if task == 'qa':
                    qa_inputs["input_ids"].append(batch["input_ids"][i].to(args.device))
                    qa_inputs["attention_mask"].append(batch["attention_mask"][i].to(args.device))
                    qa_inputs["labels"].append(batch["labels"][i].to(args.device))
                elif task == 'abstractive':
                    abstractive_inputs["input_ids"].append(batch["input_ids"][i].to(args.device))
                    abstractive_inputs["attention_mask"].append(batch["attention_mask"][i].to(args.device))
                    abstractive_inputs["labels"].append(batch["labels"][i].to(args.device))

            if qa_inputs["input_ids"]:
                qa_inputs = {k: torch.stack(v) for k, v in qa_inputs.items()}
                qa_outputs = model.generate(input_ids=qa_inputs["input_ids"], attention_mask=qa_inputs["attention_mask"], max_length=max_summary_length, task='qa')
                generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in qa_outputs]

                target_texts = []
                for t in qa_inputs["labels"]:
                    t = t[t != -100]  # Remove -100 tokens
                    target_texts.append(tokenizer.decode(t, skip_special_tokens=True))

                for g_text, t_text in zip(generated_texts, target_texts):
                    scores = rouge.get_scores(g_text, t_text)[0]
                    qa_rouge_1_scores.append(scores['rouge-1']['f'])
                    qa_rouge_2_scores.append(scores['rouge-2']['f'])
                    qa_rouge_l_scores.append(scores['rouge-l']['f'])

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
    avg_qa_rouge_1 = sum(qa_rouge_1_scores) / len(qa_rouge_1_scores) if qa_rouge_1_scores else 0
    avg_qa_rouge_2 = sum(qa_rouge_2_scores) / len(qa_rouge_2_scores) if qa_rouge_2_scores else 0
    avg_qa_rouge_l = sum(qa_rouge_l_scores) / len(qa_rouge_l_scores) if qa_rouge_l_scores else 0

    avg_abstractive_rouge_1 = sum(abstractive_rouge_1_scores) / len(abstractive_rouge_1_scores) if abstractive_rouge_1_scores else 0
    avg_abstractive_rouge_2 = sum(abstractive_rouge_2_scores) / len(abstractive_rouge_2_scores) if abstractive_rouge_2_scores else 0
    avg_abstractive_rouge_l = sum(abstractive_rouge_l_scores) / len(abstractive_rouge_l_scores) if abstractive_rouge_l_scores else 0

    logger("QA ROUGE Scores:")
    logger(f"  - ROUGE-1: {avg_qa_rouge_1}")
    logger(f"  - ROUGE-2: {avg_qa_rouge_2}")
    logger(f"  - ROUGE-L: {avg_qa_rouge_l}")

    logger("Abstractive ROUGE Scores:")
    logger(f"  - ROUGE-1: {avg_abstractive_rouge_1}")
    logger(f"  - ROUGE-2: {avg_abstractive_rouge_2}")
    logger(f"  - ROUGE-L: {avg_abstractive_rouge_l}")

    return {
        "qa": {
            "rouge-1": avg_qa_rouge_1,
            "rouge-2": avg_qa_rouge_2,
            "rouge-l": avg_qa_rouge_l
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
        scheduler, 

        qa_loss_weight, 
        summarization_loss_weight, 

        args,
        logger,

        num_epochs=20, 
        max_grad_norm=1.0, 
        eval_accumulation_steps=None,

        omega=None,
        omega_optim=None
    ):

    best_abstractive_rougel = 0
    stagnant_epochs_abstractive = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss_qa = 0.0
        total_loss_abstractive = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            optimizer_main.zero_grad()
            if omega_optim is not None: omega_optim.zero_grad()

            qa_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "decoder_attention_mask": []}
            abstractive_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "decoder_attention_mask": []}

            for i in range(len(batch['task'])):
                task = batch['task'][i]
                if task == 'qa':
                    qa_inputs["input_ids"].append(batch["input_ids"][i])
                    qa_inputs["attention_mask"].append(batch["attention_mask"][i])
                    qa_inputs["labels"].append(batch["labels"][i])
                    qa_inputs["decoder_attention_mask"].append(batch["decoder_attention_mask"][i])
                elif task == 'abstractive':
                    abstractive_inputs["input_ids"].append(batch["input_ids"][i])
                    abstractive_inputs["attention_mask"].append(batch["attention_mask"][i])
                    abstractive_inputs["labels"].append(batch["labels"][i])
                    abstractive_inputs["decoder_attention_mask"].append(batch["decoder_attention_mask"][i])
                
            qa_loss = 0
            if qa_inputs["input_ids"]:
                qa_inputs = {k: torch.stack(v).to(args.device) for k, v in qa_inputs.items()}
                qa_outputs = model(
                    input_ids=qa_inputs["input_ids"],
                    attention_mask=qa_inputs["attention_mask"],
                    labels=qa_inputs["labels"],
                    decoder_attention_mask=qa_inputs["decoder_attention_mask"],
                    task='qa'
                )
                qa_loss = qa_outputs.loss
                total_loss_qa += qa_loss.item() 

            abstractive_loss = 0
            if abstractive_inputs["input_ids"]:
                abstractive_inputs = {k: torch.stack(v).to(args.device) for k, v in abstractive_inputs.items()}
                abstractive_outputs = model(
                    input_ids=abstractive_inputs["input_ids"],
                    attention_mask=abstractive_inputs["attention_mask"],
                    labels=abstractive_inputs["labels"],
                    decoder_attention_mask=abstractive_inputs["decoder_attention_mask"],
                    task='abstractive'
                )
                abstractive_loss = abstractive_outputs.loss
                total_loss_abstractive += abstractive_loss.item()

            if omega_optim is not None:
                total_loss = (omega * abstractive_loss) + ((omega / 2) * qa_loss)
            else:    
                total_loss = (qa_loss_weight * qa_loss) + (summarization_loss_weight * abstractive_loss)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if eval_accumulation_steps is None or (step + 1) % eval_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer_main.step()
                scheduler.step()
                optimizer_main.zero_grad()

                if omega_optim is not None:
                    omega_optim.step()
                    omega_optim.zero_grad()

        rouge_scores = evaluate_model(model, valid_dataloader, tokenizer, args, logger)
        if rouge_scores is None:
            logger("Skipping evaluation due to errors.")
            continue

        current_abstractive_rougel = rouge_scores["abstractive"]["rouge-l"]
        if current_abstractive_rougel > best_abstractive_rougel:
            best_abstractive_rougel = current_abstractive_rougel
            stagnant_epochs_abstractive = 0
        else:
            stagnant_epochs_abstractive += 1

        logger(f"Epoch [{epoch+1}/{num_epochs}] - QA Loss: {total_loss_qa/len(train_dataloader):.4f} - Abstractive Loss: {total_loss_abstractive/len(train_dataloader):.4f}")
        logger("=================================\n")

        if stagnant_epochs_abstractive >= 3:
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

    logger = Logger(output_file)
    logger("CONFIGS:")
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
    # model_name = "UBC-NLP/AraT5-base"
    # model_name = "UBC-NLP/AraT5-base-title-generation"
    model_name = "UBC-NLP/AraT5v2-base-1024"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AraT5_PMTL(AutoModelForSeq2SeqLM.from_pretrained(model_name), decoder_split_level=args.decoder_split_level).to(args.device)
    # model = AraT5_PMTL(T5ForConditionalGeneration.from_pretrained(model_name, resume_download=True)).to(args.device)

    optimizer = AdamW(model.parameters(), lr = 5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=3000, num_training_steps=66000)
    # MODEL LOADING -------------------------------------

    # DATA LOADING --------------------------------------
    train_data_abs = pd.read_csv('./data/train1_processed.csv')
    valid_data_abs = pd.read_csv('./data/valid_processed.csv')
    test_data_abs= pd.read_csv('./data/test_processed.csv')
    ood_data = pd.read_csv('./data/ood_processed.csv')
    ext_data = pd.read_csv('./data/extractive_data.csv')

    train_data_ext, temp_data_ext = train_test_split(ext_data, test_size=0.1, random_state=42)
    valid_data_ext, test_data_ext = train_test_split(temp_data_ext, test_size=0.5, random_state=42)

    # drop nans
    train_data_abs = train_data_abs.dropna()
    valid_data_abs = valid_data_abs.dropna()
    test_data_abs = test_data_abs.dropna()
    ood_data = ood_data.dropna()
    train_data_ext = train_data_ext.dropna()
    valid_data_ext = valid_data_ext.dropna()
    test_data_ext = test_data_ext.dropna()

    train_dataset = MultiTaskDataset(tokenizer, qa_data=train_data_ext if args.single_task == 0 else None, summarization_data=train_data_abs)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    validation_dataset = MultiTaskDataset(tokenizer, qa_data=valid_data_ext if args.single_task == 0 else None, summarization_data=valid_data_abs)
    valid_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = MultiTaskDataset(tokenizer, qa_data=test_data_ext if args.single_task == 0 else None, summarization_data=test_data_abs)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ood_dataset = MultiTaskDataset(tokenizer, qa_data=test_data_ext if args.single_task == 0 else None, summarization_data=ood_data)
    ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger("Data loaded successfully")
    logger(f"EXTRACTIVE TRAIN DATA:{train_data_abs.shape}")
    logger(f"EXTRACTIVE VALID DATA:{valid_data_abs.shape}")
    logger(f"EXTRACTIVE TEST DATA:{test_data_abs.shape}")
    logger("\n")
    logger(f"ABSTRACTIVE TRAIN DATA:{train_data_ext.shape}")
    logger(f"ABSTRACTIVE VALID DATA:{valid_data_ext.shape}")
    logger(f"ABSTRACTIVE TEST DATA:{test_data_ext.shape}")
    logger("\n")
    logger(f"OOD DATA:{ood_data.shape}")
    # DATA LOADING --------------------------------------

    logger("\nTraining=====================================\n")

    omega, omega_optim = None, None
    if WEIGHTING_SETTING[args.weighting_setting] == "relative":
        omega = nn.Parameter(torch.tensor(1, dtype=torch.float), requires_grad=True)
        omega_optim = torch.optim.AdamW([omega], lr=5e-6)
    elif WEIGHTING_SETTING[args.weighting_setting] == "grad":
        omega = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        omega_optim = AdamW([omega], lr=5e-6)

    train(
        model, 
        train_dataloader, 
        valid_dataloader, 
        tokenizer, 
        optimizer, 
        scheduler, 
        args.abstractive_weight, 
        args.extractive_weight, 
        args,
        logger,
        num_epochs=args.epochs,
        omega=omega,
        omega_optim=omega_optim
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
    parser.add_argument('--single_task', dest='single_task', default='0')
    parser.add_argument('--weighting_setting', dest='weighting_setting', default='0')
    parser.add_argument('--abstractive_weight', dest='abstractive_weight', default='0.4')
    parser.add_argument('--extractive_weight', dest='extractive_weight', default='0.6')
    parser.add_argument('--decoder_split_level', dest='decoder_split_level', default='1')
    parser.add_argument('--batch_size', dest='batch_size', default='8')
    parser.add_argument('--epochs', dest='epochs', default='20')
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