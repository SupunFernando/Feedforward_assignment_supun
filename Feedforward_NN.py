import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
print('done')

import time
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data1=r'C:\Users\Supun\Desktop\Supun Backup\Supun\transfer\documents\myfiles\MScDA\DS\NN\assignments_NN\Assignment-01\crx.data'
cols=['cat1','con1','con2','cat2','cat3','cat4','cat5','con3','cat6','cat7','con4','cat8','cat9','con5','con6','final']
df = pd.read_csv(data1, sep=",", header=None,names=cols)
print('done')

df = df[df.cat1 != "?"]
df = df[df.cat2 != "?"]
df = df[df.cat3 != "?"]
df = df[df.cat4 != "?"]
df = df[df.cat5 != "?"]
df = df[df.cat6 != "?"]
df = df[df.cat7 != "?"]
df = df[df.cat8 != "?"]
df = df[df.cat9 != "?"]
df = df[df.con1 != "?"]
df = df[df.con5 != "?"]
print(df)

df=df.sample(frac=1)
continuous = ['con1','con2','con3','con4','con5','con6']
cs = MinMaxScaler()
cts = cs.fit_transform(df[continuous])

cat1 = LabelBinarizer().fit(df['cat1'])
cat1 = cat1.transform(df['cat1'])

cat2 = LabelBinarizer().fit(df['cat2'])
cat2 = cat2.transform(df['cat2'])

cat3 = LabelBinarizer().fit(df['cat3'])
cat3 = cat3.transform(df['cat3'])

cat4 = LabelBinarizer().fit(df['cat4'])
cat4 = cat4.transform(df['cat4'])

cat5 = LabelBinarizer().fit(df['cat5'])
cat5 = cat5.transform(df['cat5'])

cat6 = LabelBinarizer().fit(df['cat6'])
cat6 = cat6.transform(df['cat6'])

cat7 = LabelBinarizer().fit(df['cat7'])
cat7 = cat7.transform(df['cat7'])

cat8 = LabelBinarizer().fit(df['cat8'])
cat8 = cat8.transform(df['cat8'])

cat9 = LabelBinarizer().fit(df['cat9'])
cat9 = cat9.transform(df['cat9'])

final = LabelBinarizer().fit(df['final'])
final = final.transform(df['final'])

df = np.hstack([cts,cat1,cat2,cat3,cat4,cat5,cat6,cat7,cat8,cat9])

dfT=df
finalT=final

totF1=0



"""
Bin1
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(542):
    trainData.append(df[i])
    trainLabels.append(final[i])

for i in range(524,653):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1

"""
Bin2
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(393):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(524,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(393,524):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1

"""
Bin3
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(262):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(393,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(262,393):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1

"""
Bin4
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(131):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(262,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(131,262):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1

"""
Bin5
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]


for i in range(131,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(131):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    

recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1


print("AVERAGE OF F1 FOR ARCHITECTURE 1 : ",totF1/5)
f1Case1=totF1/5

"""
-------------------------------------DIFFERENT ARCHTECTURE 2 -------------------------
"""
totF1=0
trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(542):
    trainData.append(df[i])
    trainLabels.append(final[i])

for i in range(524,653):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(8, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin2
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(393):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(524,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(393,524):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(8, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin3
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(262):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(393,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(262,393):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(8, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin4
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(131):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(262,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(131,262):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(8, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin5
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]


for i in range(131,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(131):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(8, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    

recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1


print("AVERAGE OF F1 FOR ARCHITECTURE 2 : ",totF1/5)
f1Case2=totF1/5

time.sleep(5)

"""
-------------------------------------DIFFERENT ARCHTECTURE 3 -------------------------
"""
totF1=0
trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(542):
    trainData.append(df[i])
    trainLabels.append(final[i])

for i in range(524,653):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(4, input_dim=len(trainData[0]), activation="tanh"))
model.add(Dense(2, activation="sigmoid"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin2
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(393):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(524,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(393,524):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(4, input_dim=len(trainData[0]), activation="tanh"))
model.add(Dense(2, activation="sigmoid"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin3
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(262):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(393,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(262,393):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(4, input_dim=len(trainData[0]), activation="tanh"))
model.add(Dense(2, activation="sigmoid"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin4
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(131):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(262,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(131,262):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(4, input_dim=len(trainData[0]), activation="tanh"))
model.add(Dense(2, activation="sigmoid"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin5
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]


for i in range(131,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(131):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(4, input_dim=len(trainData[0]), activation="tanh"))
model.add(Dense(2, activation="sigmoid"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    

recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1


print("AVERAGE OF F1 FOR ARDGHITECTURE 3 : ",totF1/5)
f1Case3=totF1/5
time.sleep(5)

"""
-------------------------------------DIFFERENT ARCHTECTURE 4 -------------------------
"""
totF1=0
trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(542):
    trainData.append(df[i])
    trainLabels.append(final[i])

for i in range(524,653):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin2
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(393):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(524,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(393,524):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin3
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(262):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(393,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(262,393):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin4
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]

for i in range(131):
    trainData.append(df[i])
    trainLabels.append(final[i])
for i in range(262,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(131,262):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    
recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1
"""
Bin5
"""

trainData=[]
trainLabels=[]
testData=[]
testLabels=[]


for i in range(131,653):
    trainData.append(df[i])
    trainLabels.append(final[i])


for i in range(131):
    testData.append(df[i])
    testLabels.append(final[i])


trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

trainData=np.squeeze(trainData)
testData=np.squeeze(testData)

model = Sequential()
model.add(Dense(16, input_dim=len(trainData[0]), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
opt = Adam(lr=1e-3, decay=1e-3 / 40)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

model.fit(trainData, trainLabels,epochs=40, batch_size=20)    
p=model.predict(testData)
cm=confusion_matrix(testLabels.argmax(axis=1), p.argmax(axis=1))    

recall=cm[1][1]/(cm[1][1]+cm[1][0])
pre=cm[1][1]/(cm[1][1]+cm[0][1])
f1=2*(pre*recall)/(pre+recall)
totF1=totF1+f1


print("AVERAGE OF F1 FOR ARDGHITECTURE 4 : ",totF1/5)
f1Case4=totF1/5
time.sleep(5)

print("Architecture 1 F1 = ", f1Case1)
print("Architecture 2 F1 = ", f1Case2)
print("Architecture 3 F1 = ", f1Case3)
print("Architecture 4 F1 = ", f1Case4)

