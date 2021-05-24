from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# # 2.1 Входные данные
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Обучающие данные
train_images.shape
len(train_labels)
train_labels

# Контрольные данные
test_images.shape
len(test_labels)
test_labels

# 2.2 Сама СЕТЬ
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 2.3 Этап компиляции
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# 2.4 Входные данные
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 2.5 Подготовка меток
train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels,10)

# Цикл обучения
network.fit(train_images, train_labels, epochs=5, batch_size=128)
#
# Проверим, как модель распознает контрольный набор
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)
