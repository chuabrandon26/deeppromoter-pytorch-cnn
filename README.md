# deeppromoter-pytorch-cnn
A PyTorch implementation of the DeePromoter hybrid CNN-BiLSTM architecture for high-accuracy eukaryotic promoter prediction and genomic motif discovery.
# Project Overview
This project implements DeePromoter, a hybrid deep learning model designed to accurately predict eukaryotic promoter activity from DNA sequences. Identifying promoter regions is critical in bioinformatics as they regulate the initiation of gene transcription.

# The script provides a complete pipeline for:
Preprocessing DNA sequences into one-hot encoded tensors.
Training a multi-kernel CNN combined with a BiLSTM.
Evaluating performance using the Matthews Correlation Coefficient (MCC), which is more robust for genomic data than standard accuracy.

# What I Did & The Purpose
As part of my Master’s research in Molecular Bioscience (Systems Biology) at Heidelberg University, I adapted this script to:
Implement a Hybrid Architecture: I utilized a Convolutional Neural Network (CNN) for local motif extraction (like TATA boxes) and a Bidirectional Long Short-Term Memory (BiLSTM) to capture long-range sequence dependencies.
Optimize Multi-Kernel Scanning: The model uses multiple kernel sizes (27, 14, 7) to scan the sequence at different resolutions simultaneously.
​Enhanced Reliability: I integrated MCC as the primary evaluation metric to ensure the model distinguishes between real promoters and "challenging" negative sets, rather than just random genomic noise.
​Automation: I built a robust training loop with automatic checkpointing and early stopping to prevent overfitting.

# The Script
This is the core training logic used in the repository. It handles the setup of the local environment and the primary training execution.
````python
import torch
import torch.optim as optim
from torch import nn
from icecream import ic
from pathlib import Path
import os
import sys

# Step 1: Fix environment to avoid ModuleNotFound errors
if "DeePromoter" in os.listdir() and not os.getcwd().endswith("DeePromoter"):
    os.chdir("DeePromoter")

# Step 2: Ensure local modules are prioritized
sys.path.insert(0, os.getcwd())

# Import local architecture and dataloaders
from modules.deepromoter import DeePromoter
from dataloader import load_data
from model_eval import evaluate, mcc # Renamed 'test' to 'model_eval' to avoid built-in conflicts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data_path, pretrain=None, exp_name="promoter_test", training=True, ker=None, epoch_num=1000):
    if ker is None:
        ker = [27, 14, 7] # Multi-kernel sizes for CNN scanning

    # Set up output directory
    exp_folder = Path("./output").joinpath(exp_name)
    exp_folder.mkdir(parents=True, exist_ok=True)

    ic("Loading DNA sequence data...")
    data = load_data(data_path, device=device)
    train_pos, val_pos, _, train_neg, val_neg, _ = data

    # Initialize Hybrid CNN-BiLSTM Model
    model = DeePromoter(ker)
    model.to(device)

    if pretrain:
        model.load_state_dict(torch.load(pretrain))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    mcc_scores = []
    best_mcc, counter, patience = 0, 0, 10

    ic("Starting Deep Learning Training Loop")
    if training:
        for epoch in range(epoch_num):
            model.train()
            for (batch_pos, batch_neg) in zip(train_pos, train_neg):
                inputs = torch.cat((batch_pos[0], batch_neg[0]), 0)
                labels = torch.cat((batch_pos[1], batch_neg[1]), 0)

                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, labels.long())
                loss.backward()
                optimizer.step()

            # Validation phase every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    eval_data, _ = evaluate(model, [val_pos, val_neg])
                    prec, rec, current_mcc = mcc(eval_data)
                
                mcc_scores.append(current_mcc)
                ic(epoch, current_mcc, prec, rec)

                # Save the current state
                torch.save(model.state_dict(), f"{exp_folder}/epoch_{epoch}.pth")

                # Track best performance (MCC)
                if current_mcc > best_mcc:
                    best_mcc = current_mcc
                    torch.save(model.state_dict(), f"{exp_folder}/best_mcc.pth")
                    counter = 0
                else:
                    counter += 1
                
                if counter >= patience:
                    ic("Early stopping: Model converged.")
                    break
                    
    return mcc_scores
