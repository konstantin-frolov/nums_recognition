# Читаем картинку с камеры, делаем предобработку и прогоняем через нейросеть, распознающую рукописные цифры

import cv2 as cv
import numpy as np
from nnet_3layers import NNet


# Функция поиска центра контура по моментам изображения. На входе картинка, на выходе координаты центра x,y
def find_center(frame):
    M = cv.moments(frame)
    if M['m00'] != 0:
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    return cx, cy


class GiveMeVideo:
    def __init__(self):
        self.cap = cv.VideoCapture(0)

    def start_video(self):
        while True:
            self.ret, self.frame = self.cap.read()
            self.frame = cv.cvtColor(self.frame, cv.COLOR_RGB2GRAY)
            cv.imshow('Video', self.frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv.destroyAllWindows()

    def start_recognition_nums(self, NNet):
        kernel = np.ones((15, 15))
        while True:
            # Предобработка картинки Делаем чернобелой и фильтруем чтобы получить белую цифру на черном фоне
            self.ret, self.frame = self.cap.read()
            self.frame = cv.cvtColor(self.frame[0:, int(self.frame.shape[1]/2 - self.frame.shape[0]/2):
                                                    int(self.frame.shape[1]/2 + self.frame.shape[0]/2)],
                                     cv.COLOR_RGB2GRAY)
            a, frame2 = cv.threshold(self.frame, np.mean(self.frame)-20, 255, cv.THRESH_BINARY_INV)
            frame2 = cv.dilate(frame2, kernel, iterations=1)
            # стабилизируем цифру. Ищем центр цифры и сдвигаем картинку центр картинки=центр цифры
            cx, cy = find_center(frame2)
            if frame2.shape[0]/2-cx < frame2.shape[0]/2 and frame2.shape[1]/2 - cy < frame2.shape[0]/2:
                frame2 = np.roll(frame2, int(frame2.shape[0]/2)-cx, axis=1)
                frame2 = np.roll(frame2, int(frame2.shape[1]/2)-cy, axis=0)
            # Сжимаем картинку и масштабируем значения, чтобы запихнуть в нейросеть
            frame_shaped = np.reshape(cv.resize(frame2, (28, 28)), (1, 784))
            scaled_frame = np.asfarray(frame_shaped) / 255.0 * 0.99 + 0.01
            # Засовываем в сеть. Если отклик на цифру больше 0.85 выводим ее значение, иначе говорим что числа нет
            ans = NNet.query(scaled_frame)
            if max(ans) < 0.85:
                print('No nums in view')
                print(np.argmax(ans), max(ans))
            else:
                print(np.argmax(ans), max(ans))
            # Склеиваем картинку до обработки и перед сетью. Выводим на экран. Кнопка "q" выкидывает из проги
            cv.imshow('Video', np.concatenate((self.frame, frame2), axis=1))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()


# Задаем сеть, грузим в неё ранее рассчитанные веса и запускаем обработку видео сетью
num_in, num_hide, num_out = 784, 200, 10
learning_rate = 0.2
n = NNet(num_in, num_out, num_hide, learning_rate=learning_rate)
n.load_W(np.load('W_in_h.npy'), np.load('W_h_out.npy'))
video = GiveMeVideo()
video.start_recognition_nums(n)
