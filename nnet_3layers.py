# Задает структуру класса сощдания нейросети из 3 слоев (один скрытый слой)

import numpy as np
from scipy import special as sp


class NNet:
    # Инициализация сети. num_in - число входов, num_out - число выходов, num_hide - число скрытых нейронов
    # learning_rate - коэф-т обучения (по умолчанию = 0.3)
    def __init__(self, num_in, num_out, num_hide, learning_rate=0.3):
        self.in_nodes = num_in
        self.out_nodes = num_out
        self.hide_nodes = num_hide
        self.LR = learning_rate
        # Создаем матрицу весов для 3-х слоев
        self.W_in_h = np.random.normal(0.0, pow(self.hide_nodes, -0.5), (self.hide_nodes, self.in_nodes))
        self.W_h_out = np.random.normal(0.0, pow(self.out_nodes, -0.5), (self.out_nodes, self.hide_nodes))
        # Задаем активационную функцию
        self.activation_func = lambda x: sp.expit(x)
        pass

    # Если есть готовые веса для сети загружаем их
    def load_W(self, W_in_h, W_h_out):
        self.W_in_h = W_in_h
        self.W_h_out = W_h_out
        pass

    # Обучение сети на 1-ом примере. input_list - массив входов, target_list - массив ответа на пример
    def train(self, input_list, target_list):
        inputs = np.transpose(np.array(input_list, ndmin=2))
        targets = np.transpose(np.array(target_list, ndmin=2))
        hide_outs = self.activation_func(np.dot(self.W_in_h, inputs))
        outputs = self.activation_func(np.dot(self.W_h_out, hide_outs))
        err = targets - outputs
        hide_err = np.dot(np.transpose(self.W_h_out), err)
        self.W_h_out += self.LR * np.dot(err * outputs * (1 - outputs), np.transpose(hide_outs))
        self.W_in_h += self.LR * np.dot(hide_err * hide_outs * (1 - hide_outs), np.transpose(inputs))
        pass

    # Получаем веса нейросети
    def return_W(self):
        return self.W_in_h, self.W_h_out
        pass

    # Прямой проход нейросети на 1-ом примере. input_list - массив входов. outs - массив ответа сети
    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hide_outs = self.activation_func(np.dot(self.W_in_h, inputs))
        outs = self.activation_func(np.dot(self.W_h_out, hide_outs))
        return outs
        pass


