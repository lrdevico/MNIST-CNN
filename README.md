#  Convolutional Neural Network (CNN) — MNIST Classification

This project implements my **first Convolutional Neural Network (CNN)** from scratch using **PyTorch**, applied to the **MNIST handwritten digits dataset**.  
The main goal is to **strengthen my understanding of convolutional architectures** and their application to image classification tasks, while maintaining a clear and reproducible workflow.

--

## Project Overview
The workflow includes:

- Loading and preprocessing the **MNIST dataset**  
- Normalizing and reshaping image data for CNN input  
- Implementing a **custom CNN architecture** with convolutional and pooling layers  
- Training using **mini-batches** and the **Adam optimizer**  
- Evaluating model performance on a held-out test set  
- Visualizing predictions, misclassifications, and model confidence levels  

---

## Technical Details

| Component | Description |
|------------|-------------|
| **Framework** | PyTorch |
| **Dataset** | MNIST (70,000 grayscale digit images, 28×28 px) |
| **Model** | 2 Convolutional layers + 1 Fully Connected layer |
| **Loss Function** | Cross-Entropy |
| **Optimizer** | Adam |
| **Language** | Python 3 |

---

## Objectives
This project serves as both a **learning milestone** and a **reference implementation** for future deep learning experiments.  
It focuses on understanding the **end-to-end training pipeline** of CNNs — from data preprocessing and forward/backward propagation to performance evaluation and error analysis.

--

## 🇧🇷 Versão em Português

Este projeto representa minha **primeira implementação de uma Rede Neural Convolucional (CNN)** utilizando **PyTorch**, aplicada ao dataset **MNIST** de dígitos manuscritos.  
O objetivo principal é **reforçar o entendimento sobre arquiteturas convolucionais** e suas aplicações em tarefas de classificação de imagens, mantendo um fluxo de trabalho claro e reproduzível.

### Visão Geral
- Carregamento e pré-processamento do dataset MNIST  
- Normalização e ajuste das imagens para o formato aceito pela CNN  
- Implementação de uma arquitetura **customizada** com camadas convolucionais e de pooling  
- Treinamento com **mini-batches** e o otimizador **Adam**  
- Avaliação no conjunto de teste e visualização dos erros de classificação  

### Objetivos
Este repositório serve como um **marco de aprendizado** e **base de referência** para estudos futuros em deep learning.  
O foco está em compreender o pipeline completo de treinamento de uma CNN — desde o pré-processamento até a análise dos resultados.

---

## Requirements

To run the project:

```bash
pip install torch torchvision scikit-learn matplotlib numpy pandas
