import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dr = sys.argv[1]

plt.figure(1)
for root, dirs, files in os.walk(dr):
    for file in files:
        file_path = os.path.join(root, file)  # Get full file path
        data = pd.read_csv(file_path, sep='\t')
        df = pd.DataFrame(data)
        if "axi" in file:
            plt.plot(np.array(df["epoch"]), np.array(df["dice_coef"]), '-|', label=file)
        if "cor" in file:
            plt.plot(np.array(df["epoch"]), np.array(df["dice_coef"]), '-.', label=file)
        if "sag" in file:
            plt.plot(np.array(df["epoch"]), np.array(df["dice_coef"]), '-*', label=file)

plt.xlabel('Epoch')
plt.ylabel('Training Dice Coefficient')
plt.legend(title='Fold/view')
plt.title('Dice Coefficient')
plt.show()

plt.figure(2)
for root, dirs, files in os.walk(dr):
    for file in files:
        file_path = os.path.join(root, file)
        data = pd.read_csv(file_path,sep='\t')
        df = pd.DataFrame(data)
        if "axi" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["val_dice_coef"]),'-|',label=file)
        if "cor" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["val_dice_coef"]),'-.',label=file)
        if "sag" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["val_dice_coef"]),'-*',label=file)
plt.xlabel('Epoch')
plt.ylabel('Validation Dice Coefficient')
plt.legend(title='Fold/view')
plt.title('Dice Coefficient')
plt.show()

plt.figure(3)
for root, dirs, files in os.walk(dr):
    for file in files:
        file_path = os.path.join(root, file)
        data = pd.read_csv(file_path,sep='\t')
        df = pd.DataFrame(data)
        if "axi" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["loss"]),'-|',label=file)
        if "cor" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["loss"]),'-.',label=file)
        if "sag" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["loss"]),'-*',label=file)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(title='Fold/view')
plt.title('Loss')
plt.show()

plt.figure(4)
for root, dirs, files in os.walk(dr):
    for file in files:
        file_path = os.path.join(root, file)
        data = pd.read_csv(file_path,sep='\t')
        df = pd.DataFrame(data)
        if "axi" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["val_loss"]),'-|',label=file)
        if "cor" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["val_loss"]),'-.',label=file)
        if "sag" in file:
            plt.plot(np.array(df["epoch"]),np.array(df["val_loss"]),'-*',label=file)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend(title='Fold/view')
plt.title('Loss')
plt.show()

