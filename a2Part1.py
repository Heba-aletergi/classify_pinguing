import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from NeuralNetwork import Neural_Network


def encode_labels(labels):
    # encode 'Adelie' as 1, 'Chinstrap' as 2, 'Gentoo' as 3
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded

def accuracy(predictions, actual_onehot_encoded, instance_length):
    test_correct_predictions = 0
    for predicted_label, actual_label in zip(predictions, actual_onehot_encoded):
        if np.array_equal(predicted_label, actual_label):
            test_correct_predictions += 1

    test_accuracy = test_correct_predictions / instance_length
    return test_accuracy

if __name__ == '__main__':
    data = pd.read_csv('penguins307-train.csv')
    # the class label is last!
    labels = data.iloc[:, -1]
    # seperate the data from the labels
    instances = data.iloc[:, :-1]
    #scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    instances = scaler.fit_transform(instances)
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    # labels = onehot_encoded

    # Parameters. As per the handout.
    n_in = 4
    n_hidden = 2
    n_out = 3
    learning_rate = 0.2

    initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])
    initial_hidden_bias = np.array([[-0.02, -0.20]])
    initial_output_bias = np.array([[-0.33, 0.26, 0.06]])

    nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        learning_rate, initial_output_bias, initial_hidden_bias)

    print('First instance has label {}, which is {} as an integer, and {} as a list of outputs.\n'.format(
        labels[0], integer_encoded[0], onehot_encoded[0]))

    # need to wrap it into a 2D array
    instance1_prediction = nn.predict([instances[0]])
    if instance1_prediction[0] is None:
        # This should never happen once you have implemented the feedforward.
        instance1_predicted_label = "???"
    else:
        instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction)
    print('Predicted label for the first instance is: {}\n'.format(instance1_predicted_label))

    # TODO: Perform a single backpropagation pass using the first instance only. (In other words, train with 1
    #  instance for 1 epoch!). Hint: you will need to first get the weights from a forward pass.
    nn.train(instances[0:1], onehot_encoded[0], 1)

    print('Weights after performing BP for first instance only:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    # TODO: Train for 100 epochs, on all instances.
    nn.train(instances, onehot_encoded, 10)
    print('\nAfter training:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)


    pd_data_ts = pd.read_csv('penguins307-test.csv')
    test_labels = pd_data_ts.iloc[:, -1]
    test_instances = pd_data_ts.iloc[:, :-1]
    #scale the test according to our training data.
    test_instances = scaler.transform(test_instances)

    # TODO: Compute and print the test accuracy
    # Perform prediction on test data 
    tes_label_encoder, test_integer_encoded, test_onehot_encoder, test_onehot_encoded = encode_labels(test_labels)
    predictions = nn.predict(test_instances)
    
    # Calculate the accuracy
    predictions_class = onehot_encoder.transform(np.array(predictions).reshape(-1, 1))  # convert interget predicted into one hot code class
    acc_test = accuracy(predictions_class, test_onehot_encoded , len(test_instances))
    print("The Accuracy on test data is: {:.2f} %".format(acc_test*100))
