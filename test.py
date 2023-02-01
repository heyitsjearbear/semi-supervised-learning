
import numpy as np
from sklearn.model_selection import KFold
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout

def create_model():
    model = Sequential()
    model.add(Embedding(8000, 32, input_length=250))
    model.add(LSTM(20))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_new_train_data(predictions, fold_n):
    X_new = list()
    y_new = list()
    
    for i, prediction in enumerate(predictions):
        if prediction > 0.95 or prediction < 0.05:
            X_new.append(X_fold[fold_n][i])
            y_new.append(np.argmax(prediction))
    
    return np.array(X_new), np.array(y_new)

def join_shuffle(X_train, y_train, X_new, y_new):
    X_train = np.vstack((X_train, X_new))
    y_train = np.append(y_train, y_new)
    
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    return X_train[indices], y_train[indices]

(X, y), (X_test, y_test) = imdb.load_data(num_words=8000)

# Finding out the maximum length of every review.
lengths = list()
for x in X:
    lengths.append(len(x))
   
# The output is 238.71364.
print(np.mean(lengths))

# Limit all reviews to be 250 words.
X_pad = pad_sequences(X, maxlen=250)
X_test_pad = pad_sequences(X_test, maxlen=250)

kf = KFold(n_splits=5, shuffle=True)

X_fold = list()
y_fold = list()

for _, fold in kf.split(X_pad):
    
    X_fold.append(X_pad[fold])
    y_fold.append(y[fold])
    
X_fold = np.array(X_fold)
y_fold = np.array(y_fold)

# The 0th fold will be our known labeled, the rest folds are assumed to be unlabeled
X_train = X_fold[0]
y_train = y_fold[0]

model = create_model()
model.fit(X_train[:-1000], y_train[:-1000], epochs=2, 
          validation_data=(X_train[-1000:], y_train[-1000:]))
          
predictions = model.predict(X_fold[1])
X_new, y_new = get_new_train_data(predictions, 1)
X_train, y_train = join_shuffle(X_train, y_train, X_new, y_new)

model = create_model()
model.fit(X_train[:-1000], y_train[:-1000], epochs=3, 
          validation_data=(X_train[-1000:], y_train[-1000:]))

# Predict samples in fold 2
predictions = model.predict(X_fold[2])

# Filter out samples in fold 2
X_new, y_new = get_new_train_data(predictions, 2)

# Concatenate new data to X_train and y_train
X_train, y_train = join_shuffle(X_train, y_train, X_new, y_new)

model.fit(X_train, y_train, epochs=2, 
          validation_data=(X_test_pad, y_test))