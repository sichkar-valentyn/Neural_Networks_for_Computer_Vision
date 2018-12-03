# File: Solver.py
# Description: Neural Networks for computer vision in autonomous vehicles and robotics
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904




# Creating Solver class for training classification models and for predicting

# Importing needed library
import numpy as np
import pickle
from Image_Classification.Helper_Functions import optimize_rules

"""
Solver encapsulates all the logic necessary for training classification models.
It performs Stochastic Gradient Descent using different updating rules defined in
Helper_Functions/optimize_rules.py

Solver accepts both training and validation data and labels.
Consequently, it can periodically check classification accuracy on both
training and validation data to watch out for overfitting.

Firstly, for training, instance of Solver will be constructed with model of classifier,
datasets, and other options like learning rate, batch size, etc.
After that, method 'train()' will be called to run optimization procedure and train the model.

Solver works on model object that have to contain following:
    model.params has to be a dictionary mapping string parameters names
    to numpy arrays with values.

    model.loss_for_training(x, y) has to be a functions that computes loss and gradients.
    Loss will be a scalar and gradients will be a dictionary with the same keys
    as in model.params mapping parameters names to gradients with respect to those parameters.

    model.scores_for_predicting(x) has to be a function that computes classification scores.
    Scores will be an array of shape (N, C) giving classification scores for x,
    where scores[i, c] gives the score of class c for x[i].


"""


# Creating class for Solver
class Solver(object):

    """""""""
    Initializing new Solver instance
    Input consists of following required and Optional arguments.
    
    Required arguments consist of following:
        model - a modal object conforming parameters as described above,
        data - a dictionary with training and validating data.
    
    Optional arguments (**kwargs) consist of following:
        update_rule - a string giving the name of an update rule in optimize_rules.py,
        optimization_config - a dictionary containing hyperparameters that will be passed 
                              to the chosen update rule. Each update rule requires different
                              parameters, but all update rules require a 'learning_rate' parameter.
        learning_rate_decay - a scalar for learning rate decay. After each epoch the 'learning_rate'
                              is multiplied by this value,
        batch_size - size of minibatches used to compute loss and gradients during training,
        number_of_epochs - the number of epoch to run for during training,
        print_every - integer number that corresponds to printing loss every 'print_every' iterations,
        verbose_mode - boolean that corresponds to condition whether to print details or not. 

    """

    def __init__(self, model, data, **kwargs):
        # Preparing required arguments
        self.model = model
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_validation = data['x_validation']
        self.y_validation = data['y_validation']

        # Preparing optional arguments
        # Unpacking keywords of arguments
        # Using 'pop' method and setting at the same time default value
        self.update_rule = kwargs.pop('update_rule', 'sgd')  # Default is 'sgd'
        self.optimization_config = kwargs.pop('optimization_config', {})  # Default is '{}'
        self.learning_rate_decay = kwargs.pop('learning_rate_decay', 1.0)  # Default is '1.0'
        self.batch_size = kwargs.pop('batch_size', 100)  # Default is '100'
        self.number_of_epochs = kwargs.pop('number_of_epochs', 10)  # Default is '10'
        self.print_every = kwargs.pop('print_every', 10)  # Default is '10'
        self.verbose_mode = kwargs.pop('verbose_mode', True)  # Default is 'True'

        # Checking if there are extra keyword arguments and raising an error
        if len(kwargs) > 0:
            extra = ', '.join(k for k in kwargs.keys())
            raise ValueError('Extra argument:', extra)

        # Checking if update rule exists and raising an error if not
        # Using function 'hasattr(object, name)',
        # where 'object' is our imported module 'optimize_rules'
        # and 'name' is the name of the function inside
        if not hasattr(optimize_rules, self.update_rule):
            raise ValueError('Update rule', self.update_rule, 'does not exists')

        # Reassigning string 'self.update_rule' with the real function from 'optimize_rules'
        # Using function 'getattr(object, name)',
        # where 'object' is our imported module 'optimize_rules'
        # and 'name' is the name of the function inside
        self.update_rule = getattr(optimize_rules, self.update_rule)

        # Implementing '_reset' function
        self._reset()

    # Creating 'reset' function for defining variables for optimization
    def _reset(self):
        # Setting up variables
        self.current_epoch = 0
        self.best_validation_accuracy = 0
        self.best_params = {}
        self.loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

        # Making deep copy of 'optimization_config' for every parameter at every layer
        # It means that at least learning rate will be for every parameter at every layer
        self.optimization_configurations = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.optimization_configurations[p] = d

    # Creating function 'step' for making single gradient update
    def _step(self):
        # Making minibatch from training data
        # Getting total number of training images
        number_of_training_images = self.x_train.shape[0]
        # Getting random batch of 'batch_size' size from total number of training images
        batch_mask = np.random.choice(number_of_training_images, self.batch_size)
        # Getting training dataset according to the 'batch_mask'
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Calculating loss and gradient for current minibatch
        loss, gradient = self.model.loss_for_training(x_batch, y_batch)

        # Adding calculated loss to the history
        self.loss_history.append(loss)

        # Implementing updating for all parameters (weights and biases)
        # Going through all parameters
        for p, v in self.model.params.items():
            # Taking current value of derivative for current parameter
            dw = gradient[p]
            # Defining configuration for current parameter
            config_for_current_p = self.optimization_configurations[p]
            # Implementing updating and getting next values
            next_w, next_configuration = self.update_rule(v, dw, config_for_current_p)
            # Updating value in 'params'
            self.model.params[p] = next_w
            # Updating value in 'optimization_configurations'
            self.optimization_configurations[p] = next_configuration

    # Creating function for checking accuracy of the model on the current provided data
    # Accuracy will be used in 'train' function for both training dataset and for testing dataset
    # Depending on which input into the model will be provided
    def check_accuracy(self, x, y, number_of_samples=None, batch_size=100):

        """""""""
        Input consists of following:
            x of shape (N, C, H, W) - N data, each with C channels, height H and width W,
            y - vector of labels of shape (N,),
            number_of_samples - subsample data and test model only on this number of data,
            batch_size - split x and y into batches of this size to avoid using too much memory.

        Function returns:
            accuracy - scalar number giving percentage of images 
                       that were correctly classified by model.
        """

        # Getting number of input images
        N = x.shape[0]

        # Subsample data if 'number_of_samples' is not None
        # and number of input images is more than 'number_of_samples'
        if number_of_samples is not None and N > number_of_samples:
            # Getting random batch of 'number_of_samples' size from total number of input images
            batch_mask = np.random.choice(N, number_of_samples)
            # Reassigning (decreasing) N to 'number_of_samples'
            N = number_of_samples
            # Getting dataset for checking accuracy according to the 'batch_mask'
            x = x[batch_mask]
            y = y[batch_mask]

        # Defining and calculating number of batches
        # Also, making it as integer with 'int()'
        number_of_batches = int(N / batch_size)
        # Increasing number of batches if there is no exact match of input images over 'batch_size'
        if N % batch_size != 0:
            number_of_batches += 1

        # Defining variable for storing predicted class for appropriate input image
        y_predicted = []

        # Computing predictions in batches
        # Going through all batches defined by 'number_of_batches'
        for i in range(number_of_batches):
            # Defining start index and end index for current batch of images
            s = i * batch_size
            e = (i + 1) * batch_size
            # Getting scores by calling function 'loss_for predicting' from model
            scores = self.model.scores_for_predicting(x[s:e])
            # Appending result to the list 'y_predicted'
            # Scores is given for each image with 10 numbers of predictions for each class
            # Getting only one class for each image with maximum value
            y_predicted.append(np.argmax(scores, axis=1))
            # Example
            #
            # a = np.arange(6).reshape(2, 3)
            # print(a)
            #    ([[0, 1, 2],
            #     [3, 4, 5]])
            #
            # print(np.argmax(a))
            # 5
            #
            # np.argmax(a, axis=0)
            #     ([1, 1, 1])
            #
            # np.argmax(a, axis=1)
            #     ([2, 2])
            #
            # Now we have each image with its only one predicted class (index of each row)
            # but not with 10 numbers for each class

        # Concatenating list of lists and making it as numpy array
        y_predicted = np.hstack(y_predicted)

        # Finally, we compare predicted class with correct class for all input images
        # And calculating mean value among all values of following numpy array
        # By saying 'y_predicted == y' we create numpy array with True and False values
        # 'np.mean' function will return average of the array elements
        # The average is taken over the flattened array by default
        accuracy = np.mean(y_predicted == y)

        # Returning accuracy
        return accuracy

    # Creating function for training the model
    def train(self):
        # Getting total number of training images
        number_of_training_images = self.x_train.shape[0]
        # Calculating number of iterations per one epoch
        # If 'number_of_training_images' is less than 'self.batch_size' then we chose '1'
        iterations_per_one_epoch = max(number_of_training_images / self.batch_size, 1)
        # Calculating total number of iterations for all process of training
        # Also, making it as integer with 'int()'
        iterations_total = int(self.number_of_epochs * iterations_per_one_epoch)

        # Running training process in the loop for total number of iterations
        for t in range(iterations_total):
            # Making single step for updating all parameters
            self._step()

            # Checking if training loss has to be print every 'print_every' iteration
            if self.verbose_mode and t % self.print_every == 0:
                # Printing current iteration and showing total number of iterations
                # Printing currently saved loss from loss history
                print('Iteration: ' + str(t + 1) + '/' + str(iterations_total) + ',',
                      'loss =', self.loss_history[-1])

            # Defining variable for checking end of current epoch
            end_of_current_epoch = (t + 1) % iterations_per_one_epoch == 0

            # Checking if it is the end of current epoch
            if end_of_current_epoch:
                # Incrementing epoch counter
                self.current_epoch += 1
                # Decaying learning rate for every parameter at every layer
                for k in self.optimization_configurations:
                    self.optimization_configurations[k]['learning_rate'] *= self.learning_rate_decay

            # Defining variables for first and last iterations
            first_iteration = (t == 0)
            last_iteration = (t == iterations_total - 1)

            # Checking training and validation accuracy
            # At the first iteration, the last iteration, and at the end of every epoch
            if first_iteration or last_iteration or end_of_current_epoch:
                # Checking training accuracy with 1000 samples
                training_accuracy = self.check_accuracy(self.x_train, self.y_train,
                                                        number_of_samples=1000)

                # Checking validation accuracy
                # We don't specify number of samples as it has only 1000 images itself
                validation_accuracy = self.check_accuracy(self.x_validation, self.y_validation)

                # Adding calculated accuracy to the history
                self.train_accuracy_history.append(training_accuracy)
                self.validation_accuracy_history.append(validation_accuracy)

                # Checking if the 'verbose_mode' is 'True' then printing details
                if self.verbose_mode:
                    # Printing current epoch over total amount of epochs
                    # And training and validation accuracy
                    print('Epoch: ' + str(self.current_epoch) + '/' + str(self.number_of_epochs) + ',',
                          'Training accuracy = ' + str(training_accuracy) + ',',
                          'Validation accuracy = ' + str(validation_accuracy))

                # Tracking the best model parameters by comparing validation accuracy
                if validation_accuracy > self.best_validation_accuracy:
                    # Assigning current validation accuracy to the best validation accuracy
                    self.best_validation_accuracy = validation_accuracy
                    # Reset 'self.best_params' dictionary
                    self.best_params = {}
                    # Assigning current parameters to the best parameters variable
                    for k, v in self.model.params.items():
                        self.best_params[k] = v

        # At the end of training process swapping best parameters to the model
        self.model.params = self.best_params

        # Saving trained model parameters into 'pickle' file
        with open('Serialized_Models/model_params_ConvNet1.pickle', 'wb') as f:
            pickle.dump(self.model.params, f)

        # Saving loss, training accuracy and validation accuracy histories into 'pickle' file
        history_dictionary = {'loss_history': self.loss_history,
                              'train_accuracy_history': self.train_accuracy_history,
                              'validation_history': self.validation_accuracy_history}
        with open('Serialized_Models/model_histories_ConvNet1.pickle', 'wb') as f:
            pickle.dump(history_dictionary, f)
