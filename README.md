# PSI5120 - Tópicos em Computação em Nuvem (2023)

![Badge development](http://img.shields.io/static/v1?label=states&message=%20Full&color=blue&style=for-the-badge)

![System Architecture](https://github.com/ruan-math/Rede_Neural_MLP/blob/main/cloud.jpg)

## Rede_Neural_MLP

[Link do Relatório Final ](https://github.com/ruan-math/Rede_Neural_MLP/blob/main/Trabalho%20final.pdf)

### Explorando o Poder da Computação em Nuvem para o Aprendizado de máquina usando uma rede neural  MLP (Multi Layer Perceptron) para classificação de objetos em imagens Reais

Repositório do Código em Python de Aprendizado de Maquina Usando uma Rede Neural MLP (Multi Layer Perceptron) para 
classificação de objetos em imagens.

[Link do ppt da Apresentação do trabalho final ](https://github.com/ruan-math/Rede_Neural_MLP/blob/main/Computa%C3%A7%C3%A3o%20em%20nuvem%20para%20aprendizado%20de%20m%C3%A1quina.pdf)

# Implementação do Codigo de Treinameto da Rede MLP.

Perceptron Multicamadas (PMC ou MLP — Multi Layer Perceptron) é uma rede neural com uma ou mais camadas ocultas com um número indeterminado de neurônios. A camada oculta possui esse nome porque não é possível prever a saída desejada nas camadas intermediárias.

Para treinar a rede MLP, o algoritmo comumente utilizado é o de retropropagação (Backpropagation) seu arquitetura comforme a representação abaixo:

![System Architecture](https://github.com/ruan-math/Rede_Neural_MLP/blob/main/MLP.png)


É uma aplicação de técnicas de aprendizado de máquina para resolver um problema de classificação de imagens. O conjunto de dados Fashion MNIST consiste em 60.000 imagens de 10 categorias diferentes de roupas, com 6.000 imagens por categoria. Cada imagem é uma representação em escala de cinza de 28x28 pixels.

# Implementação da Rede Neural MLP
``````
import os
import tensorflow.keras as keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import sys
``````

# Carregamento dos dados
``````
(AX, AY), (QX, QY) = fashion_mnist.load_data()
AX = 255 - AX
QX = 255 - QX

nclasses = 10
AY2 = to_categorical(AY, nclasses)
QY2 = to_categorical(QY, nclasses)

nl, nc = AX.shape[1], AX.shape[2]  # 28, 28
AX = AX.astype('float32') / 255.0  # 0 a 1
QX = QX.astype('float32') / 255.0  # 0 a 1
``````

# Definição do modelo
``````
model = Sequential()
model.add(Flatten(input_shape=(nl, nc)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(nclasses, activation='softmax'))
``````

# Resumo do modelo
``````
model.summary()
``````

# Compilação do modelo
``````
opt = optimizers.Adam()
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
``````

# Treinamento do modelo com registro de métricas
``````
history = model.fit(AX, AY2,
                    batch_size=100,
                    epochs=100,
                    verbose=True,
                    validation_data=(QX, QY2))
``````

# Plotagem dos gráficos
``````
plt.figure(figsize=(12, 4))
``````

# Gráfico da perda
``````
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
``````

# Gráfico da acurácia
``````
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
``````

# Gráfico do erro
``````
plt.subplot(1, 3, 3)
plt.plot(history.history['accuracy'], label='Train Error')
plt.plot(history.history['val_accuracy'], label='Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.show()
``````

# Avaliação final do modelo
``````
score = model.evaluate(QX, QY2, verbose=False)
print('Test loss:', score[0])
print('Test accuracy: %.2f %%' % (100 * score[1]))
print('Test error: %.2f %%' % (100 * (1 - score[1])))
``````

# Salvando o modelo treinado

``````
model.save('MLP4.h5')
``````








Dica: Aumentar o número de camadas e neurônios nem sempre é a melhor solução para uma melhoria de performance/acurácia.

Na verdade uma das limitações da rede MLP é que ao se aumentar muito o número de camadas e neurônios ela tende a ficar com um número de parâmetros muito grande e com isso tão pesada ao ponto do hardware não conseguir processar e ela não convergir (chegar a um resultado), talvez por essa razão, até a evolução do hardware ela tenha ficado um pouco estagnada.

