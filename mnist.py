# %%

from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%

# carregando o dataset MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.to_numpy()
y = y.to_numpy()
# %%

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

# %%

# alterando o shape de um sample para matriz 28x28 para visualização
single_sample = X[0, :].reshape(28, 28) # primeiro sample --> pega todos os pixels --> reshape para matriz 28x28
print(f'Single sample shape: {single_sample.shape}')
# %%
# Plot single sample (M x N matrix)
plt.gray()
plt.matshow(single_sample)
plt.show()
# %%

#transformando o formato das imagens de (70000, 784) para (70 000, 1, 28, 28)
X = X / 255.0  # normalizando os valores dos pixels para o intervalo [0, 1]
X = X.reshape(-1, 1, 28, 28)  # reshape para (n_samples, n_channels, height, width)

# %%
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.astype(np.int64))
# %%

# dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42, stratify=y)
# %%

# arquitetura da cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)  # 1 canal de entrada, 8 filtros, kernel 3x3
        self.conv2 = nn.Conv2d(8, 16, 3) # 16 filtros, kernel 3x3
        self.fc1 = nn.Linear(16*5*5, 10) # output 10 classes (dígitos 0-9) --> 16*5*5 é o tamanho do tensor após as camadas convolucionais e pooling
    def forward(self, x):
        x = F.relu(self.conv1(x))          # conv1 + ReLU
        x = F.max_pool2d(x, 2)             # MaxPooling 2x2
        x = F.relu(self.conv2(x))          # conv2 + ReLU
        x = F.max_pool2d(x, 2)             # MaxPooling 2x2
        x = x.view(x.size(0), -1)          # Flatten
        x = self.fc1(x)                    # Fully connected
        return x

# %%

model = CNN()

# loss e optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #model.parameters() retorna os pesos e viés do modelo, lr é a learning rate
# adam é um otimizador, ou seja, um algoritmo que atualiza os pesos do modelo com base na loss calculada


# criando batches de forma manual
batch_size = 64
num_epochs = 5

for epoch in range(num_epochs): # loop de epochs
    permutation = torch.randperm(X_train.size(0)) # embaralha os indices dos samples
    epoch_loss = 0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size] # pega os indices para o batch atual
        batch_X, batch_y = X_train[indices], y_train[indices] # cria o batch
        
        optimizer.zero_grad()                  # zera os gradientes para o batch
        outputs = model(batch_X)               # forward pass
        loss = criterion(outputs, batch_y)     # calcula loss
        loss.backward()                        # backpropagation
        optimizer.step()                       # atualiza pesos
        
        epoch_loss += loss.item()              # acumula a loss do epoch
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(range(0, X_train.size(0), batch_size)):.4f}")

# %%
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
# %%

num_samples = 5
indices = np.random.choice(len(X_test), num_samples, replace=False)

for idx in indices:
    img = X_test[idx]          # (1, 28, 28)
    label_true = y_test[idx]   # label real

    with torch.no_grad():
        output = model(img.unsqueeze(0))  # adiciona batch dim → (1,1,28,28)
        pred = torch.argmax(output, dim=1).item()  # previsão do modelo

    plt.imshow(img.squeeze(), cmap='gray')  # remove canal extra para plotar
    plt.title(f"Label real: {label_true}, Predição: {pred}")
    plt.axis('off')
    plt.show()
# %%
import torch
import matplotlib.pyplot as plt

model.eval()

# todas as previsões
with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)

# Índices onde o modelo errou
errors = (preds != y_test).nonzero(as_tuple=True)[0]

print(f"Total de erros: {len(errors)}")
print(f"Acurácia: {(1 - len(errors)/len(y_test))*100:.2f}%")

# plotando alguns erros
num_to_show = min(50, len(errors))

plt.figure(figsize=(12, 12))
for i, idx in enumerate(errors[:num_to_show]):
    img = X_test[idx]
    true_label = y_test[idx].item()
    pred_label = preds[idx].item()

    plt.subplot(5, 10, i+1)  # 5x10 grid
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"R: {true_label} / P: {pred_label}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

# %%
# pega uma amostra aleatória dos erros e mostra o quao confiante o modelo estava na predição errada
random_state = check_random_state(3)
sampled_errors = random_state.choice(errors.numpy(), size=min(10, len(errors)), replace=False)

for idx in sampled_errors:
    img = X_test[idx]
    true_label = y_test[idx].item()
    pred_label = preds[idx].item()
    confidence = torch.softmax(outputs[idx], dim=0)[pred_label].item()

    plt.subplot(5, 10, i+1)  # 5x10 grid
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"R: {true_label} / P: {pred_label} / Conf: {confidence:.2f}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
# %%
