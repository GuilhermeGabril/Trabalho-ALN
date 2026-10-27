# Projeto de Reconhecimento Facial com Haar Cascade e LBPH

## Descrição do Projeto

Este projeto está sendo desenvolvido como parte da disciplina **Álgebra Linear Numérica**, com o objetivo de aplicar conceitos teóricos em um problema prático de reconhecimento facial.

O sistema utiliza a biblioteca **OpenCV**, aplicando o algoritmo **Haar Cascade** para a detecção de rostos e o **LBPH (Local Binary Pattern Histogram)** para o reconhecimento facial.

Essa abordagem permite trabalhar com matrizes, extração de características e cálculos de distâncias métricas, relacionando-se aos conceitos matemáticos estudados em aula.

Será utilizado um **dataset fornecido pelo professor** para o treinamento e a avaliação do sistema.

---

## Integrantes do Grupo

- Guilherme Gabril
- Sheldron Grandini
- Geliel Galhardo

---

## Objetivo

Desenvolver um sistema capaz de detectar, reconhecer e comparar rostos, aplicando conceitos de Álgebra Linear Numérica.

---

## Tecnologias Utilizadas

- Python
- NumPy
- OpenCV
- Haar Cascade
- LBPH

---

## Como Executar

### 1. Instalar as Dependências

```bash
pip install numpy opencv-contrib-python
```

### 2. Treinar o Modelo

Execute o arquivo `treinar.py` para treinar o modelo utilizando as imagens do dataset:

```bash
python treinar.py
```

### 3. Testar o Modelo

Após concluir o treinamento, execute o arquivo `testar.py` para realizar os testes e visualizar os resultados:

```bash
python testar.py
```

**Importante:** execute primeiro o treinamento e depois os testes.
