import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import csv


from sklearn.impute import SimpleImputer

def load_data(train_files, test_file):
    # train_data = []
    # for file in train_files:
    #     train_data.append(pd.read_csv(file, sep='\t'))
    # train_data = pd.concat(train_data)

    train_data = pd.read_csv(train_files)
    train_data = train_data.head(20000)
    test_data = pd.read_csv(test_file)
    test_data = test_data.head(20000)

    categorical_features = train_data.columns[2:33]
    binary_features = train_data.columns[33:42]
    numerical_features = train_data.columns[42:80]

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')

    X_categorical = encoder.fit_transform(train_data[categorical_features])
    X_binary = train_data[binary_features].values

    # # ##
    # print("Numerical features shape:", train_data[numerical_features].shape)
    # print("Sample numerical features:\n", train_data[numerical_features].head())

    # # ##
    X_numerical = scaler.fit_transform(imputer.fit_transform(train_data[numerical_features]))

    X_train = np.hstack([X_categorical, X_binary, X_numerical])
    y_train = train_data[['is_clicked', 'is_installed']].values

    X_categorical_test = encoder.transform(test_data[categorical_features])
    X_binary_test = test_data[binary_features].values
    X_numerical_test = scaler.transform(imputer.transform(test_data[numerical_features]))
    

    X_test = np.hstack([X_categorical_test, X_binary_test, X_numerical_test])

    return X_train, y_train, X_test, test_data


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    optimizer = Adam(lr=0.001)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch > 75:
        lr = initial_lr * 0.01
    elif epoch > 50:
        lr = initial_lr * 0.1
    else:
        lr = initial_lr
    return lr

def main():
    train_files = 'data//data2.csv'  # Update with the training data file path
    test_file = 'toy_test.csv'  # Update with the testing data file path

    X_train, y_train, X_test, test_data = load_data(train_files, test_file)

    model = create_model(X_train.shape[1])

    lr_scheduler = LearningRateScheduler(lr_schedule)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model.fit(X_train_split, y_train_split,
              batch_size=32,
              epochs=100,
              validation_data=(X_val_split, y_val_split),
              shuffle=True,
              callbacks=[lr_scheduler])

    y_pred = model.predict(X_test)

    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['RowId', 'is_clicked', 'is_installed'])
        for row_id, (is_clicked, is_installed) in zip(test_data['f_0'], y_pred):
            writer.writerow([row_id, is_clicked, is_installed])


if __name__ == '__main__':
    main()
