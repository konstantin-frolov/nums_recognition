# Функция задает тренировку и тест для сети, по какому файлу MNIST и количество эпох обучения
# train_nnet - вход: указатель класса NNet, путь к файлу для тренировки и кол-во эпох обучения (по умолчанию = 2)
#              выход: ничего, внутри функции обновятся веса нейросети
# test_nnet - вход: указатель класса NNet, путь к файлу для тестирования сети
#             выход: вероятность правильного распознавания


def train_nnet(nnet, train_file, epochs=2):
    import numpy as np
    file_data = open(train_file, 'r')
    data_train = file_data.readlines()
    file_data.close()
    for epoch in range(epochs):
        for record in data_train:
            values = record.split(',')
            scaled_input = np.asfarray(values[1:]) / 255.0 * 0.99 + 0.01
            targets = np.zeros(nnet.out_nodes) + 0.01
            targets[int(values[0])] = 0.99
            nnet.train(scaled_input, targets)
            pass


def test_nnet(nnet, test_file):
    import numpy as np
    file_test = open(test_file, 'r')
    data_test = file_test.readlines()
    file_test.close()
    store_card = []
    for record in data_test:
        values = record.split(',')
        correct_label = int(values[0])
        scaled_input = np.asfarray(values[1:]) / 255.0 * 0.99 + 0.01
        outputs = nnet.query(scaled_input)
        label = np.argmax(outputs)
        if label == correct_label:
            store_card.append(1)
        else:
            store_card.append(0)
            pass
        pass
    efficiency = sum(store_card[:]) / len(store_card)
    return efficiency
