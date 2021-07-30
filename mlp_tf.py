import numpy as np
from random import random
import tensorflow as tf 
from sklearn.model_selection import train_test_split

def generate_dataset(num_samples, test_size):
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_dataset(5000, 0.3)
    
    # build model: 2 inputs, 5 hidden layers, 1 output
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10
        , input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="MSE")

    # train the model
    model.fit(X_train, y_train, epochs=100)

    # evaluate
    print("\nModel evaluation: ")
    model.evaluate(X_test, y_test, verbose=1)

    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)
    
    print(predictions)