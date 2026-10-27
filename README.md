# Projeto de Reconhecimento Facial com Haar Cascade e LBPH

## Descrição do Projeto

Este projeto está sendo desenvolvido como parte da disciplina **Álgebra Linear Numérica**, com o objetivo de aplicar conceitos teóricos em um problema prático de reconhecimento facial.

O sistema utiliza o **Haar Cascade** para detecção de rostos e o **LBPH (Local Binary Pattern Histogram)** para reconhecimento facial, utilizando a biblioteca OpenCV.

## Integrantes do Grupo

- Guilherme Gabril
- Sheldron Grandini
- Geliel Galhardo

## Tecnologias Utilizadas

- Python
- NumPy
- OpenCV
- Haar Cascade
- LBPH

## Como Executar

### 1. Instalar as dependências

```bash
pip install numpy opencv-contrib-python
```

### 2. Treinar o modelo

Execute o comando:

```bash
python treinar.py
```

O programa utilizará as imagens de `recdev-master/treino/hard` para treinar o modelo.

Ao finalizar, serão gerados:
- `modelo_lbph.yml`: modelo treinado.
- `mapeamento_ids.txt`: quantidade de fotos utilizadas para cada pessoa.
- Pasta `rostos-processados`: imagens dos rostos extraídos durante o treinamento.

### 3. Testar o modelo

Após o treinamento, execute:

```bash
python testar.py
```

No menu apresentado, escolha:

- **1:** testar o modelo com as imagens de `recdev-master/teste/hard`.
- **2:** reconhecer uma imagem específica, informando o caminho do arquivo.
- **3:** sair do programa.

Ao escolher a opção 1, o sistema apresenta os acertos, erros, taxas de acerto e erro, além da distância retornada pelo LBPH para cada imagem.

Os resultados detalhados são salvos no arquivo `relatorio_teste.txt`.

**Importante:** execute primeiro `treinar.py` e depois `testar.py`. O dataset e o arquivo `haarcascade_frontalface_default.xml` devem estar nos caminhos esperados pelo código.
