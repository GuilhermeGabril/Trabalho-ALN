import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTE_DIR = os.path.join(BASE_DIR, "recdev-master", "teste", "easy")
XML_FILE = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODELO_FILE = os.path.join(BASE_DIR, "modelo_lbph.yml")

print("Carregando modelo...")
recognizer = cv2.face.LBPHFaceRecognizer_create()

try:
    recognizer.read(MODELO_FILE)
    print("Modelo carregado!\n")
except:
    print("Erro: Execute primeiro o 'treinar.py'")
    exit()

face_cascade = cv2.CascadeClassifier(XML_FILE)

def extrair_id(nome_arquivo):
    nome_sem_extensao = os.path.splitext(nome_arquivo)[0]
    partes = nome_sem_extensao.split('-')
    if len(partes) >= 1:
        try:
            return int(partes[0])
        except ValueError:
            return None
    return None

def testar_modelo():
    print("=" * 60)
    print("TESTANDO O MODELO COM FOTOS NOVAS")
    print("=" * 60)
    
    arquivos = [f for f in os.listdir(TESTE_DIR) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(arquivos) == 0:
        print(f"Nenhum arquivo em '{TESTE_DIR}'!")
        return
    
    print(f"Testando {len(arquivos)} fotos...\n")
    
    acertos = 0
    erros = 0
    resultados = []
    
    for arquivo in arquivos:
        pessoa_id_real = extrair_id(arquivo)
        
        if pessoa_id_real is None:
            print(f"Pulando: {arquivo}")
            continue
        
        caminho_imagem = os.path.join(TESTE_DIR, arquivo)
        img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        rostos = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        
        if len(rostos) == 0:
            print(f"{arquivo} -> Nenhum rosto detectado!")
            continue
        
        (x, y, w, h) = rostos[0]
        rosto = img[y:y+h, x:x+w]
        rosto = cv2.resize(rosto, (200, 200))
        
        label_predito, confianca = recognizer.predict(rosto)
        
        if label_predito == pessoa_id_real:
            resultado = "ACERTOU"
            acertos += 1
        else:
            resultado = f"ERROU (predito: {label_predito})"
            erros += 1
        
        resultados.append({
            'arquivo': arquivo,
            'real': pessoa_id_real,
            'predito': label_predito,
            'confianca': confianca,
            'resultado': 'Acertou' if label_predito == pessoa_id_real else 'Errou'
        })
        
        print(f"{resultado} - {arquivo} | Real: {pessoa_id_real} | Predito: {label_predito} | Conf: {confianca:.0f}")
    
    print("\n" + "=" * 60)
    print("RESULTADOS DO TESTE")
    print("=" * 60)
    print(f"Acertos: {acertos}")
    print(f"Erros: {erros}")
    print(f"Taxa de acerto: {acertos/(acertos+erros)*100:.1f}%")
    
    with open("relatorio_teste.txt", "w", encoding="utf-8") as f:
        f.write("=== RELATORIO DE TESTE ===\n\n")
        f.write(f"Total de testes: {acertos+erros}\n")
        f.write(f"Acertos: {acertos}\n")
        f.write(f"Erros: {erros}\n")
        f.write(f"Taxa de acerto: {acertos/(acertos+erros)*100:.1f}%\n\n")
        f.write("Detalhamento:\n")
        for r in resultados:
            f.write(f"{r['resultado']} - {r['arquivo']} | Real: {r['real']} | Predito: {r['predito']} | Conf: {r['confianca']:.0f}\n")
    
    print(f"\nRelatorio salvo em 'relatorio_teste.txt'")

def reconhecer_face(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    if img is None:
        print("Nao foi possivel ler a imagem!")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostos = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(rostos) == 0:
        print("Nenhum rosto encontrado!")
        return
    
    for (x, y, w, h) in rostos:
        rosto = gray[y:y+h, x:x+w]
        rosto = cv2.resize(rosto, (200, 200))
        
        label, confianca = recognizer.predict(rosto)
        
        if confianca < 100:
            texto = f"ID: {label} ({confianca:.0f})"
            cor = (0, 255, 0)
        elif confianca < 200:
            texto = f"ID {label}? ({confianca:.0f})"
            cor = (0, 255, 255)
        else:
            texto = "Desconhecido"
            cor = (0, 0, 255)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(img, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    
    cv2.imshow("Reconhecimento", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SISTEMA DE RECONHECIMENTO FACIAL")
    print("=" * 60)
    print("1 - Testar modelo com fotos novas (recdev-master/teste/easy)")
    print("2 - Reconhecer uma foto especifica")
    print("3 - Sair")
    
    opcao = input("\nEscolha: ")
    
    if opcao == "1":
        testar_modelo()
    elif opcao == "2":
        caminho = input("Caminho da foto: ")
        reconhecer_face(caminho)
    else:
        print("Saindo...")