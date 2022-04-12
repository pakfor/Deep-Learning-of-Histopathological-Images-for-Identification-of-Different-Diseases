import pandas as pd
import os

# Define formulas for calculating sensitivity, specificity, negative predictive value (NPV) and positive predictive value (PPV)
def sensitivity(TP,FN):
    return TP / (TP + FN)

def specificity(TN,FP):
    return TN / (TN + FP)

def npv(FN,TN):
    return TN / (FN + TN)

def ppv(TP,FP):
    return TP / (TP + FP)

# Define formula for calculating area under the curve (AUC)
def auc(sen,spe):
    area = 0
    for i in range(0,len(sen)-1):
        area += ((sen[i] + sen[i+1]) * abs(spe[i+1] - spe[i])) / 2
    return area

# Basic information
geneConsidered = ['EGFR','KRAS','STK11']
datasets = ['TCGA-LUAD','CPTAC-LUAD']

# Directory that holds the results from interested models
resultFileDir = 'TCGA-LUAD/Model Performance'
resultFilesAll = os.listdir(resultFileDir)

# Select data that are relevant to interested dataset
datasetShowResult = 'CPTAC-LUAD'
resultFiles = [i for i in resultFilesAll if datasetShowResult in i]

# Define thresholds to be tested
thresholds = [0,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.995,0.999,0.9995,0.9999,0.99995,0.99999,1]

# For later-on plotting individual curve for each model
modelOrder = []

# For storing sensitivity recorded at each threshold
allSenEGFR = []
allSenKRAS = []
allSenSTK = []

# For storing specificity recorded at each threshold
allSpeEGFR = []
allSpeKRAS = []
allSpeSTK = []

# Loop through different result files (loop through different trained models)
for i in range(0,len(resultFiles)):
    originalFile = pd.read_csv(f'{resultFileDir}/{resultFiles[i]}')
    model_name = resultFiles[i].split('_')[1]
    modelOrder.append(model_name)
    
    senEGFR = [1]
    senKRAS = [1]
    senSTK = [1]
    
    speEGFR = [1]
    speKRAS = [1]
    speSTK = [1]
    
    # Calculate false positive (FP), false negative (FN), true positive (TP) and true negative (TN) at each threshold level
    for threshold in thresholds:
        file = originalFile.copy()
        file['EGFR'].values[file['EGFR'].values >= threshold] = 1
        file['EGFR'].values[file['EGFR'].values < threshold] = 0
        file['KRAS'].values[file['KRAS'].values >= threshold] = 1
        file['KRAS'].values[file['KRAS'].values < threshold] = 0
        file['STK11'].values[file['STK11'].values >= threshold] = 1
        file['STK11'].values[file['STK11'].values < threshold] = 0

        EGFRPred = tuple(file['EGFR'])
        groundTruth = tuple(file['Ground Truth'])
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for i in range(0,len(EGFRPred)):
            if (EGFRPred[i] == 1 and groundTruth[i] == 0):
                TP += 1
            if (EGFRPred[i] == 0 and groundTruth[i] != 0):
                TN += 1
            if (EGFRPred[i] == 1 and groundTruth[i] != 0):
                FP += 1
            if (EGFRPred[i] == 0 and groundTruth[i] == 0):
                FN += 1
        
        print('Result for EGFR:')
        print('TP:',TP)
        print('TN:',TN)
        print('FP:',FP)
        print('FN:',FN)
        print('Sensitivity:',sensitivity(TP,FN))
        print('Specificity:',specificity(TN,FP))
        try:
            print('Positive Predictive Value (PPV):',ppv(TP,FP))
            print('Negative Predictive Value (NPV):',npv(FN,TN))
        except:
            print('Error')
        
        senEGFR.append(sensitivity(TP,FN))
        speEGFR.append(1-specificity(TN,FP))
        
        KRASPred = tuple(file['KRAS'])
        groundTruth = tuple(file['Ground Truth'])
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for i in range(0,len(KRASPred)):
            if (KRASPred[i] == 1 and groundTruth[i] == 1):
                TP += 1
            if (KRASPred[i] == 0 and groundTruth[i] != 1):
                TN += 1
            if (KRASPred[i] == 1 and groundTruth[i] != 1):
                FP += 1
            if (KRASPred[i] == 0 and groundTruth[i] == 1):
                FN += 1
        
        print('Result for KRAS:')
        print('TP:',TP)
        print('TN:',TN)
        print('FP:',FP)
        print('FN:',FN)
        print('Sensitivity:',sensitivity(TP,FN))
        print('Specificity:',specificity(TN,FP))
        try:
            print('Positive Predictive Value (PPV):',ppv(TP,FP))
            print('Negative Predictive Value (NPV):',npv(FN,TN))
        except:
            print('Error')
        
        senKRAS.append(sensitivity(TP,FN))
        speKRAS.append(1-specificity(TN,FP))
        
        STK11Pred = tuple(file['STK11'])
        groundTruth = tuple(file['Ground Truth'])
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for i in range(0,len(STK11Pred)):
            if (STK11Pred[i] == 1 and groundTruth[i] == 2):
                TP += 1
            if (STK11Pred[i] == 0 and groundTruth[i] != 2):
                TN += 1
            if (STK11Pred[i] == 1 and groundTruth[i] != 2):
                FP += 1
            if (STK11Pred[i] == 0 and groundTruth[i] == 2):
                FN += 1
        
        print('Result for STK11:')
        print('TP:',TP)
        print('TN:',TN)
        print('FP:',FP)
        print('FN:',FN)
        print('Sensitivity:',sensitivity(TP,FN))
        print('Specificity:',specificity(TN,FP))
        try:
            print('Positive Predictive Value (PPV):',ppv(TP,FP))
            print('Negative Predictive Value (NPV):',npv(FN,TN))
        except:
            print('Error')
        
        senSTK.append(sensitivity(TP,FN))
        speSTK.append(1-specificity(TN,FP))
    
    senEGFR.append(0)
    senKRAS.append(0)
    senSTK.append(0)
    
    speEGFR.append(0)
    speKRAS.append(0)
    speSTK.append(0)
    
    allSenEGFR.append(senEGFR)
    allSenKRAS.append(senKRAS)
    allSenSTK.append(senSTK)
    
    allSpeEGFR.append(speEGFR)
    allSpeKRAS.append(speKRAS)
    allSpeSTK.append(speSTK)

#%%

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300

numModel = 4

# Also include the performance of a random classifier as baseline
modelOrder.append('Random Classifier (AUC = 0.5)')

# Plot AUCs according to gene

EGFRModelOrder = []

for i in range(0,numModel):
    plt.plot(allSpeEGFR[i],allSenEGFR[i])
    EGFRModelOrder.append(modelOrder[i] + f' (AUC = {str(round(auc(allSenEGFR[i],allSpeEGFR[i]),4))})')

EGFRModelOrder.append('Random Classifier (AUC = 0.5)')
plt.plot([0,1],[0,1],'--')
plt.legend(EGFRModelOrder,loc=4,prop={'size': 9})
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.show()

KRASModelOrder = []

for i in range(0,numModel):
    plt.plot(allSpeKRAS[i],allSenKRAS[i])
    KRASModelOrder.append(modelOrder[i] + f' (AUC = {str(round(auc(allSenKRAS[i],allSpeKRAS[i]),4))})')

KRASModelOrder.append('Random Classifier (AUC = 0.5)')
plt.plot([0,1],[0,1],'--')
plt.legend(KRASModelOrder,loc=4,prop={'size': 9})
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.show()

STKModelOrder = []

for i in range(0,numModel):
    plt.plot(allSpeSTK[i],allSenSTK[i])
    STKModelOrder.append(modelOrder[i] + f' (AUC = {str(round(auc(allSenSTK[i],allSpeSTK[i]),4))})')

STKModelOrder.append('Random Classifier (AUC = 0.5)')
plt.plot([0,1],[0,1],'--')
plt.legend(STKModelOrder,loc=4,prop={'size': 9})
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.show()