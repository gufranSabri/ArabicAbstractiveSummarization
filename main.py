import os
import warnings
import json
from tqdm import tqdm
from rouge import Rouge
import pandas as pd
import numpy as np
import random
import argparse
import datetime
from scheduler import LinearDecayLR

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer, AdamW
from sklearn.model_selection import train_test_split

from data import MultiTaskDataset
from model import *
from logger import Logger


WEIGHTING_SETTING = {
    "0": "static",
    "1": "relative",
    "2": "grad"
}

def evaluate_model(model, dataloader, tokenizer, args, logger, min_summary_length=5, max_summary_length=40, run_dir=None, epoch_num=None):
    model.eval()
    rouge = Rouge()

    task2_rouge_1_scores = []
    task2_rouge_2_scores = []
    task2_rouge_l_scores = []

    abstractive_rouge_1_scores = []
    abstractive_rouge_2_scores = []
    abstractive_rouge_l_scores = []

    # Prepare prediction logging
    prediction_log_path = None
    if run_dir and epoch_num is not None:
        prediction_log_path = os.path.join(run_dir, f"epoch_{epoch_num}.txt")
        prediction_log_file = open(prediction_log_path, "w", encoding="utf-8")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
            task2_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            abstractive_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            task2_indices, abstractive_indices = [], []

            for i in range(len(batch['task'])):
                task = batch['task'][i]
                if task == 'task2':
                    task2_inputs["input_ids"].append(batch["input_ids"][i].to(args.device))
                    task2_inputs["attention_mask"].append(batch["attention_mask"][i].to(args.device))
                    task2_inputs["labels"].append(batch["labels"][i].to(args.device))
                    task2_indices.append(i)
                elif task == 'abstractive':
                    abstractive_inputs["input_ids"].append(batch["input_ids"][i].to(args.device))
                    abstractive_inputs["attention_mask"].append(batch["attention_mask"][i].to(args.device))
                    abstractive_inputs["labels"].append(batch["labels"][i].to(args.device))
                    abstractive_indices.append(i)

            def decode_predictions(inputs, task_name, indices):
                nonlocal prediction_log_file
                if not inputs["input_ids"]:
                    return [], []

                inputs = {k: torch.stack(v) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    min_length=min_summary_length,
                    max_length=max_summary_length,
                    task=task_name,
                    num_beams=3,
                    repetition_penalty=3.0,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3
                )
                generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]

                target_texts = []
                for t in inputs["labels"]:
                    t = t[t != -100]  # Remove -100 tokens
                    target_texts.append(tokenizer.decode(t, skip_special_tokens=True))

                for i, (g_text, t_text) in enumerate(zip(generated_texts, target_texts)):
                    if prediction_log_file:
                        prediction_log_file.write(f"[{task_name.upper()}] Sample {indices[i]}:\n")
                        prediction_log_file.write(f"Prediction: {g_text}\n")
                        prediction_log_file.write(f"Target:     {t_text}\n")
                        prediction_log_file.write("\n")

                return generated_texts, target_texts

            # Evaluate and log
            task2_preds, task2_targets = decode_predictions(task2_inputs, 'task2', task2_indices)
            for g_text, t_text in zip(task2_preds, task2_targets):
                scores = rouge.get_scores(g_text, t_text)[0]
                task2_rouge_1_scores.append(scores['rouge-1']['f'])
                task2_rouge_2_scores.append(scores['rouge-2']['f'])
                task2_rouge_l_scores.append(scores['rouge-l']['f'])

            abstractive_preds, abstractive_targets = decode_predictions(abstractive_inputs, 'abstractive', abstractive_indices)
            for g_text, t_text in zip(abstractive_preds, abstractive_targets):
                scores = rouge.get_scores(g_text, t_text)[0]
                abstractive_rouge_1_scores.append(scores['rouge-1']['f'])
                abstractive_rouge_2_scores.append(scores['rouge-2']['f'])
                abstractive_rouge_l_scores.append(scores['rouge-l']['f'])

    if prediction_log_file:
        prediction_log_file.close()

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
        optimizer_weights=None,

        run_dir=None
    ):

    best_abstractive_rougel = 0
    stagnant_epochs_abstractive = 0
    avg_grad_norms = torch.zeros(2).to(args.device)
    lr_scheduler=LinearDecayLR(optimizer_main, num_epochs, 15)

    for epoch in range(num_epochs):
        model.train()
        total_loss_task2 = 0.0
        total_loss_abstractive = 0.0

        for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)):
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

            optimizer_main.step()
            optimizer_main.zero_grad()

            if omega_optim is not None:
                omega_optim.step()
                omega_optim.zero_grad()

        rouge_scores = evaluate_model(model, valid_dataloader, tokenizer, args, logger, run_dir=run_dir, epoch_num=args.epochs)
        if rouge_scores is None:
            logger("Skipping evaluation due to errors.")
            continue

        lr_scheduler.step()

        current_abstractive_rougel = rouge_scores["abstractive"]["rouge-l"]
        if current_abstractive_rougel > best_abstractive_rougel:
            best_abstractive_rougel = current_abstractive_rougel
            stagnant_epochs_abstractive = 0
            torch.save(run_dir, f"model.pth")
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

    # FOR MAC --------------------------------------------
    if args.device == "mps":
        torch.mps.manual_seed(seed)
        torch.backends.mps.deterministic = True
        torch.backends.mps.benchmark = False

    # FOR WINDOWS AND LINUX ------------------------------
    if args.device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === SETUP RUN DIRECTORY =============================
    if args.mode == 'train':
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = f"./outputs/{args.model}_{date}"
        os.makedirs(run_dir, exist_ok=True)

        # Save configuration
        config_dict = vars(args).copy()
        config_dict["WEIGHTING_SETTING"] = WEIGHTING_SETTING[args.weighting_setting]
        config_dict["date"] = date
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    elif args.mode == 'val':
        assert args.eval_path is not None, "Please provide --eval_path for validation mode"
        run_dir = args.eval_path
        config_path = os.path.join(run_dir, "config.json")
        assert os.path.exists(config_path), f"{config_path} does not exist"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            setattr(args, k, v)

    # Setup logger
    log_path = os.path.join(run_dir, "log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = Logger(log_path)

    # === LOG CONFIG ======================================
    model_name = "UBC-NLP/AraT5-base"
    logger("CONFIGS:")
    for k, v in vars(args).items():
        logger(f"{k}: {v}")
    logger("===============================================\n")

    # === MODEL LOADING ===================================
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AraT5_PMTL(T5ForConditionalGeneration.from_pretrained(model_name, resume_download=True)).to(args.device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # === DATA LOADING ====================================
    train_data_abs = pd.read_csv('./data/train1_processed.csv').dropna()
    valid_data_abs = pd.read_csv('./data/valid_processed.csv').dropna()
    test_data_abs = pd.read_csv('./data/test_processed.csv').dropna()
    ood_data = pd.read_csv('./data/ood_processed.csv').dropna()
    task2_data = pd.read_csv('./data/extractive_data.csv').dropna()

    train_data_task2, temp_data_task2 = train_test_split(task2_data, test_size=0.1, random_state=42)
    valid_data_task2, test_data_task2 = train_test_split(temp_data_task2, test_size=0.5, random_state=42)

    train_dataset = MultiTaskDataset(tokenizer, task2_data=train_data_task2 if args.single_task == 0 else None, summarization_data=train_data_abs)
    valid_dataset = MultiTaskDataset(tokenizer, task2_data=valid_data_task2, summarization_data=valid_data_abs)
    test_dataset = MultiTaskDataset(tokenizer, task2_data=test_data_task2, summarization_data=test_data_abs)
    ood_dataset = MultiTaskDataset(tokenizer, task2_data=None, summarization_data=ood_data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False)

    logger("Data loaded successfully")
    logger(f"EXTRACTIVE TRAIN DATA: {train_data_abs.shape}")
    logger(f"EXTRACTIVE VALID DATA: {valid_data_abs.shape}")
    logger(f"EXTRACTIVE TEST DATA: {test_data_abs.shape}\n")
    logger(f"ABSTRACTIVE TRAIN DATA: {train_data_task2.shape}")
    logger(f"ABSTRACTIVE VALID DATA: {valid_data_task2.shape}")
    logger(f"ABSTRACTIVE TEST DATA: {test_data_task2.shape}\n")
    logger(f"OOD DATA: {ood_data.shape}")

    # === TRAIN OR VALIDATE ===============================
    omega, omega_optim = None, None
    weights, optimizer_weights = None, None

    if WEIGHTING_SETTING[args.weighting_setting] == "relative":
        omega = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=True)
        omega_optim = torch.optim.AdamW([omega], lr=5e-6)

    elif WEIGHTING_SETTING[args.weighting_setting] == "grad":
        weights = torch.nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        optimizer_weights = torch.optim.AdamW([weights], lr=5e-6)

    if args.mode == 'train':
        logger("\nTraining=====================================\n")
        train(
            model,
            train_dataloader,
            valid_dataloader,
            tokenizer,
            optimizer,
            args.abstractive_weight,
            args.task2_weight,
            args,
            logger,
            num_epochs=args.epochs,
            omega=omega,
            omega_optim=omega_optim,
            weights=weights,
            optimizer_weights=optimizer_weights,
            run_dir=run_dir  # pass directory to save best model
        )

    # Load best checkpoint
    ckpt_path = os.path.join(run_dir, "model.pth")
    msg = model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    logger(f"Model loaded from {ckpt_path} with msg: {msg}")

    # === EVALUATE ========================================
    logger("\n========================================\n")
    logger("Evaluating on Test Data")
    evaluate_model(model, test_dataloader, tokenizer, args, logger, run_dir=run_dir, epoch_num=99999)

    logger("\n========================================\n")
    logger("Evaluating on OOD Data")
    evaluate_model(model, ood_dataloader, tokenizer, args, logger, run_dir=run_dir, epoch_num=99999)


    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', default='parallel')
    parser.add_argument('--mode', dest='mode', default='train')
    parser.add_argument('--eval_path', dest='eval_path', default='')
    parser.add_argument('--single_task', dest='single_task', default='1')
    parser.add_argument('--weighting_setting', dest='weighting_setting', default='0')
    parser.add_argument('--abstractive_weight', dest='abstractive_weight', default='1.0')
    parser.add_argument('--task2_weight', dest='task2_weight', default='1.0')
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
    args.task2_weight = float(args.task2_weight)

    assert os.path.exists(args.eval_path) or args.mode == 'train', "Evaluation path does not exist. Please provide a valid path or run in training mode."

    if not torch.cuda.is_available():
        args.device='mps'

    if not os.path.exists("./models"):
        os.makedirs("./models")

    main(args)