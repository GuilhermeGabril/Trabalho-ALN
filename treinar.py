import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TREINO_DIR = os.path.join(BASE_DIR, "recdev-master", "treino", "easy")
XML_FILE = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(XML_FILE)

faces = []
labels = []
contagem_por_pessoa = {}

def extrair_id(nome_arquivo):
    nome_sem_extensao = os.path.splitext(nome_arquivo)[0]
    partes = nome_sem_extensao.split('-')
    if len(partes) >= 1:
        try:
            return int(partes[0])
        except ValueError:
            return None
    return None

print("Lendo fotos de TREINO...")
print("=" * 50)

arquivos = [f for f in os.listdir(TREINO_DIR) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(arquivos) == 0:
    print(f"Nenhum arquivo encontrado em '{TREINO_DIR}'!")
    exit()

print(f"Encontrados {len(arquivos)} arquivos para treino.\n")

for arquivo in arquivos:
    pessoa_id = extrair_id(arquivo)
    
    if pessoa_id is None:
        print(f"Pulando: {arquivo}")
        continue
    
    caminho_imagem = os.path.join(TREINO_DIR, arquivo)
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Nao foi possivel ler: {arquivo}")
        continue
    
    rostos = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    
    if len(rostos) == 0:
        print(f"Nenhum rosto em: {arquivo}")
        continue
    
    (x, y, w, h) = rostos[0]
    rosto = img[y:y+h, x:x+w]
    rosto = cv2.resize(rosto, (200, 200))
    
    faces.append(rosto)
    labels.append(pessoa_id)
    
    if pessoa_id not in contagem_por_pessoa:
        contagem_por_pessoa[pessoa_id] = 0
    contagem_por_pessoa[pessoa_id] += 1
    
    print(f"{arquivo} -> Pessoa ID: {pessoa_id} (TREINO)")

print("\n" + "=" * 50)
print(f"Treinando com {len(faces)} rostos de {len(contagem_por_pessoa)} pessoas...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("modelo_lbph.yml")

print("Modelo salvo como 'modelo_lbph.yml'")

with open("mapeamento_ids.txt", "w", encoding="utf-8") as f:
    f.write("=== PESSOAS CADASTRADAS NO TREINO ===\n")
    for pessoa_id in sorted(contagem_por_pessoa.keys()):
        f.write(f"ID {pessoa_id}: {contagem_por_pessoa[pessoa_id]} fotos\n")

print("\nTREINAMENTO CONCLUIDO!")
print(f"Pessoas: {len(contagem_por_pessoa)}")
print(f"Total de rostos: {len(faces)}")