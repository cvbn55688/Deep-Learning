import random
import math

def random_weights(input_size, output_size):
    return [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(input_size)]

def random_biases_weights(size):
    return [random.uniform(-0.1, 0.1) for _ in range(size)]

def mse_function(outputs, expects):
    output_num = len(outputs)
    error_sum = 0
    for i in range(output_num):
        error_sum += (outputs[i] - expects[i]) ** 2
    return error_sum / output_num

def mse_derivative(outputs, expects):
    output_num = len(outputs)
    losses = []
    for i in range(output_num):
        gradient = (2 / output_num) * (outputs[i] - expects[i])
        losses.append(gradient)
    return losses

def binary_cross_entropy_function(outputs, expects):
    output_num = len(outputs)
    error_sum = 0
    for i in range(output_num):
        error_sum += expects[i] * math.log(outputs[i]) + (1 - expects[i]) * math.log(1 - outputs[i])
    return -error_sum

def binary_cross_entropy_derivative(outputs, expects):
    output_num = len(outputs)
    losses = []
    for i in range(output_num):
        losses.append(-(expects[i] / outputs[i]) + ((1 - expects[i]) / (1 - outputs[i])))
    return losses

def categorical_cross_entropy_function(outputs, expects):
    output_num = len(outputs)
    error_sum = 0
    for i in range(output_num):
        error_sum += expects[i] * math.log(outputs[i])
    return -error_sum
