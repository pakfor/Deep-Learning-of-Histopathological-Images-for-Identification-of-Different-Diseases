import pandas as pd

def sensitivity(TP,FN):
    return TP / (TP + FN)

def specificity(TN,FP):
    return TN / (TN + FP)

def npv(FN,TN):
    return TN / (FN + TN)

def ppv(TP,FP):
    return TP / (TP + FP)

def auc(sen,spe):
    area = 0
    for i in range(0,len(sen)-1):
        area += ((sen[i] + sen[i+1]) * abs(spe[i+1] - spe[i])) / 2
    return area
    
resultFile = 'Camelyon16/Testing/Testing Results/Model_UNet/results.csv'
result = pd.read_csv(resultFile)

thresholdList = [0,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.995,0.999,0.9995,0.9999,0.99995,0.99999,1]

f1_threshold = [0]
f1_sen = [1]
f1_ppv = [0]
allf1 = [0]

allSen = [1]
allSpe = [1]

for threshold in thresholdList:
    resultThreshold = result.copy()
    resultThreshold['Prediction Averaged Tile Value'].values[resultThreshold['Prediction Averaged Tile Value'].values >= threshold] = 1
    resultThreshold['Prediction Averaged Tile Value'].values[resultThreshold['Prediction Averaged Tile Value'].values < threshold] = 0
    print(resultThreshold['Prediction Averaged Tile Value'])
    
    resultThreshold['Ground-truth Averaged Tile Value'].values[resultThreshold['Ground-truth Averaged Tile Value'].values >= 0.5] = 1
    resultThreshold['Ground-truth Averaged Tile Value'].values[resultThreshold['Ground-truth Averaged Tile Value'].values < 0.5] = 0
    
    pred = tuple(resultThreshold['Prediction Averaged Tile Value'])
    groundTruth = tuple(resultThreshold['Ground-truth Averaged Tile Value'])
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(0,len(pred)):
        if (pred[i] == 1 and groundTruth[i] == 1):
            TP += 1
        if (pred[i] == 0 and groundTruth[i] == 0):
            TN += 1
        if (pred[i] == 1 and groundTruth[i] == 0):
            FP += 1
        if (pred[i] == 0 and groundTruth[i] == 1):
            FN += 1
    
    print('Threshold:',threshold)
    print('TP:',TP)
    print('TN:',TN)
    print('FP:',FP)
    print('FN:',FN)
    print('Sensitivity:',sensitivity(TP,FN))
    print('Specificity:',specificity(TN,FP))
    try:
        print('Positive Predictive Value (PPV):',ppv(TP,FP))
        print('F1 Score:',2 / (1 / sensitivity(TP,FN) + 1 / ppv(TP,FP)))
        
        allf1.append(2 / (1 / sensitivity(TP,FN) + 1 / ppv(TP,FP)))
        f1_sen.append(sensitivity(TP,FN))
        f1_ppv.append(ppv(TP,FP))
        f1_threshold.append(threshold)
    except:
        print('PPV: Division by 0')
    
    try:
        print('Negative Predictive Value (NPV):',npv(FN,TN))
    except:
        print('NPV: Division by 0')
    
    allSen.append(sensitivity(TP,FN))
    allSpe.append(1-specificity(TN,FP))

allSen.append(0)
allSpe.append(0)

f1_threshold.append(1)
f1_sen.append(0)
f1_ppv.append(1)
allf1.append(0)

#%% Plot ROC

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300

plt.plot(allSpe,allSen)
plt.plot([0,1],[0,1],'--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
#plt.text(0.4,0.1,'Area under the curve (AUC) = ' + str(round(auc(allSen,allSpe),4)))
plt.legend([f'Model 2 (U-Net) (AUC = {str(round(auc(allSen,allSpe),4))})','Random Classifier (AUC = 0.5000)'])
plt.show()
print('EGFR AUC:',auc(allSen,allSpe))

#%% Plot F1-threshold map

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300

plt.plot(f1_threshold,allf1)
plt.plot(f1_threshold,f1_sen)
plt.plot(f1_threshold,f1_ppv)
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('F1-Score, Precision, Recall Curves')
plt.xlabel('Threshold')
plt.ylabel('F1-Score, Precision, Recall')
plt.legend(['F1-Score','Recall','Precision'])
#plt.text(0.4,0.1,'Area under the curve (AUC) = ' + str(round(auc(allSen,allSpe),4)))
plt.show()

print(max(allf1))