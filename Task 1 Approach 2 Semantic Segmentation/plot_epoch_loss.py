import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set DPI for the plot
matplotlib.rcParams['figure.dpi'] = 300

# Read excel file that contains training information, i.e., epoch, training loss, testing loss
resultXLSX = pd.read_excel('E:/Camelyon16 Segmentation/Trained Models/Selected/20220401_UNet_SN_Tr5000_Te1000_EP100_FP16_Adam/log/training_activities.xlsx')

# Extract tuple that contains relevant information from the loaded excel file
epoch = tuple(resultXLSX['Epoch'])
train_loss = tuple(resultXLSX['Training Loss'])
test_loss = tuple(resultXLSX['Testing Loss'])

# Plot the training loss-epoch and testing loss-epoch in the same graph
plt.plot(epoch,train_loss)
plt.plot(epoch,test_loss)
plt.title('Loss-Epoch Plot')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Loss)')
plt.legend(['Training Loss','Testing Loss'])
plt.show()

