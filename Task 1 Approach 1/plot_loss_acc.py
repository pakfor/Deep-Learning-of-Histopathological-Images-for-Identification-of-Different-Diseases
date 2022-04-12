import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300

# Load file (in excel) that contains training activities
resultXLSX = pd.read_excel('Camelyon16/Training/Training Results/Model_InceptionV3/log/training_activities.xlsx') # training_activities.xlsx shall be created by user through recording the training loss, testing loss, training accuracy, testing accuracy at each epoch during the training 

epoch = tuple(resultXLSX['Epoch'])
train_loss = tuple(resultXLSX['Training Loss'])
train_acc = tuple(resultXLSX['Training Accuracy'])
test_loss = tuple(resultXLSX['Testing Loss'])
test_acc = tuple(resultXLSX['Testing Accuracy'])

# Plot loss against epoch
plt.plot(epoch,train_loss)
plt.plot(epoch,test_loss)
plt.title('Loss-Epoch Plot')
plt.xlabel('Epoch')
plt.ylabel('Loss (Categorical Cross-entropy)')
plt.legend(['Training Loss','Testing Loss'])
plt.show()

# Plot accuracy against epoch
plt.plot(epoch,train_acc)
plt.plot(epoch,test_acc)
plt.title('Accuracy-Epoch Plot')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (No. Correctly Predicted / No. All Prediction)')
plt.legend(['Training Accuracy','Testing Accuracy'])
plt.show()
