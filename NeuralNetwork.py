import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate, hidden_bias_weights=[0,0], output_bias_weights=[0,0,0]):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        self.hidden_bias = hidden_bias_weights
        self.output_bias = output_bias_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def Actication_Func(self, input):
        activation = 1 / (1 + np.exp(-input))  # TODO!
        return activation
    
    def _derivative(self, x):
        x_derivative = x * (1 - x)
        return x_derivative

    def softmax(self, x):
        if len(x.shape) > 1:
        # Reshape 1D array to 2D array with one column
            x = x.reshape(1, -1)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def square_error(self, y_actual, y_predicted):
        return np.mean((y_actual - y_predicted) ** 2)
    
    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        # HIDDEN LAYER --------------------------
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = np.dot(inputs, self.hidden_layer_weights[:,i]) 
            output = self.Actication_Func(weighted_sum)    
            hidden_layer_outputs.append(output)

        # OUTPUT LAYER 
        output_layer_outputs = []
        temp_output = []
        for i in range(self.num_outputs):
            # TODO! Calculate the weighted sum, and then compute the final output.  
            weighted_sum = np.dot(np.array(hidden_layer_outputs), self.output_layer_weights[:,i]) 
            temp_output.append(weighted_sum)  
        softmax_out = self.softmax(np.array(temp_output))
        output_layer_outputs.append(softmax_out)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):
        
        # Output Layer 
        output_layer_betas = np.zeros(self.num_outputs)
        # TODO! Calculate output layer betas.
        E_dout = self.square_error(desired_outputs[0], output_layer_outputs[0])
        deriv_out = self._derivative(output_layer_outputs[0])
        delta_out = E_dout * deriv_out
        output_layer_betas = delta_out
        print('OL betas: ', output_layer_betas)

        # Hidden Layer 
        hidden_layer_betas = np.zeros(self.num_hidden)
        # TODO! Calculate hidden layer betas.
        E_hidden = np.dot(delta_out, self.output_layer_weights.T)
        deriv_hidden = self._derivative(np.array(hidden_layer_outputs))
        delta_hidden = E_hidden * deriv_hidden
        hidden_layer_betas = delta_hidden
        print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # TODO! Calculate output layer weight changes.
        delta_H6_output_weights = np.dot(np.array(hidden_layer_outputs[0]), delta_out)
        delta_H6_output_weights = np.dot(np.array(hidden_layer_outputs[1]), delta_out)
        delta_output_layer_weights = np.array([delta_H6_output_weights, delta_H6_output_weights])

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # TODO! Calculate hidden layer weight changes.
        delta_input_H5_weights = np.dot(inputs.T, np.array(delta_hidden[0]))
        delta_input_H6_weights = np.dot(inputs.T, np.array(delta_hidden[1]))
        delta_hidden_layer_weights = np.array([[x,y] for x,y in zip(delta_input_H5_weights,delta_input_H6_weights)])

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        # TODO! Update the weights.
        self.output_layer_weights -= self.learning_rate * delta_output_layer_weights
        self.hidden_layer_weights -= self.learning_rate * delta_hidden_layer_weights
        #print('Placeholder')

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            correct_predictions = 0
            for i, instance in enumerate(instances):
                desired_y = [desired_outputs if len(desired_outputs.shape) == 1 else desired_outputs[i]]
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_y)
                predicted_class = np.argmax(output_layer_outputs[0]) 
                # Check if prediction correct or not 
                if predicted_class == np.argmax(desired_y[0]):
                    correct_predictions += 1

                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            print('Hidden layer weights \n', self.hidden_layer_weights)
            print('Output layer weights  \n', self.output_layer_weights)

            # TODO: Print accuracy achieved over this epoch
            acc = correct_predictions / len(instances)
            print("The Accuracy is: {:.2f} %".format(acc*100))
            

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            #print(output_layer_outputs)
            predicted_class = np.argmax(output_layer_outputs[0])  # TODO! Should be 0, 1, or 2.
            predictions.append(predicted_class)
        return predictions