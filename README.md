# Digit Recognizer

In this notebook, our goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. Simple logistic regression is not a good way to learn complex nonlinear hypotheses. We'll ultilizy keras for build a neural network that can read de digitis, we can use other techniques instead of deep learning but here we use deep learn. Neural Networks are a pretty old algorithm that was originally motivated by the goal of having machines that can mimic the brain, 
Dataset: https://www.kaggle.com/c/digit-recognizer/data  
It is not our goal to develop and optimize the ranking model. The example has a merely didactic purpose.

We start analysing the data.



```python
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import classification_report, confusion_matrix



train_df = pd.read_csv('/home/leandro/Desktop/Pytorch/train.csv')
test = pd.read_csv('/home/leandro/Desktop/Pytorch/test.csv')


x = train_df.iloc[:,1:]
y = train_df.iloc[:,0]

fig, ax = plt.subplots(1,1, figsize=(10,7))
sns.countplot(x='label', data=train_df, ax=ax)
ax.patch.set_alpha(0)

fig.text(0.1,0.92,"Distribution by Label in Mnist", fontweight="bold", fontfamily='serif', fontsize=17)


print('Data Shape')
print(train_df.shape)
print(test.shape)



```

![data](https://user-images.githubusercontent.com/83521233/124021752-4d60bb80-d9c2-11eb-9a2a-9c800541c8ad.png)

![Distribution](https://user-images.githubusercontent.com/83521233/124021738-489c0780-d9c2-11eb-9c45-8bf9ee44f0fa.png)


```python
fig , axes = plt.subplots(5, 5, figsize=(15,15))

x_idx = 0
y_idx = 0

for i in range(5*5):
    if x_idx == 5:
        x_idx = 0
        y_idx += 1
        
    axes[y_idx][x_idx].imshow(x[i].reshape(28,28), 'gray')
    axes[y_idx][x_idx].axis("off")
    x_idx += 1

plt.show()


```

![Numbers](https://user-images.githubusercontent.com/83521233/124021839-6bc6b700-d9c2-11eb-8aba-2a710d775fc8.png)

We'll construct a simple neural network, with one hidden layer and 10 outputs. We don't will split the training data set, we'll train our model with all the data we have and test him with the test data set submiting him in kaggle.



```python
#Prepare data

length = x.shape[0]
x = x.reshape(-1, 28*28)
test = test.values.reshape(-1, 28*28)
x = x.astype('float32')/  255
y = keras.utils.to_categorical(y, 10)
```


```python
#Neural network

model = Sequential()
model.add(Dense(800, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


batch_size = 32
epochs = 30
history = model.fit(x, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x, y))

predictions = model.predict_classes(test, verbose=0)

batch_size = 32
epochs = 30
history = model.fit(x, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x, y))

predictions = model.predict_classes(test, verbose=0)


```

Now our model is trained let's plot the confusion matrix.


```python

#Plot the confusion matrix. Set Normalize = True/False
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Classificando toda base de teste
y_pred = model.predict_classes(x)
# voltando pro formato de classes
y_test_c = np.argmax(y, axis=1)
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#Confution Matrix
cm = confusion_matrix(y_test_c, y_pred)
plot_confusion_matrix(cm, target_names, normalize=False, title='Confusion Matrix')
plt.show()

```

https://user-images.githubusercontent.com/83521233/124022337-0c1cdb80-d9c3-11eb-8c0a-a36bbc0623a0.png

Now we label our test dataset with our predicted results


```python
# submission
submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
    "Label": predictions})
submissions.to_csv("/home/leandro/Desktop/Pytorch/DR.csv", index=False, header=True)
```

In our first try we have  a score of 0.97892.
