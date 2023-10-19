import sys
import timeit
import cv2
import numpy as np
import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks



####################################################

INPUT_IMAGE = '114.bmp'
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 100

#################################################################

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''
    img_out = np.where(img > 0, -1, 0)
    label = 1
    aux = []
    height = img.shape[0]
    width = img.shape[1]

    tblr = {
        'T': -1,
        'B': -1,
        'L': -1,
        'R': -1,
        'label': 1,
        'n_pixels': 0
    }
    count = 0
    component = []
    for y in range(0, height):
        for x in range(0, width):
            if img_out[y][x] == -1:
                componentCopy = flood_fill(img_out, x, y, height, width, tblr, True)
                if componentCopy['n_pixels'] >= n_pixels_min and componentCopy['B'] - componentCopy['T'] >= altura_min \
                        and componentCopy['R'] - componentCopy['L'] >= largura_min:
                    component.append(componentCopy.copy())

                tblr['label'] += 1
                tblr['n_pixels'] = 0

    return component


    #===============================================================================

def compara_tblr(tblrAntigo, tblrNovo):
    if tblrNovo['L'] < tblrAntigo['L']:
        tblrAntigo['L'] = tblrNovo['L']

    if tblrNovo['T'] < tblrAntigo['T']:
        tblrAntigo['T'] = tblrNovo['T']

    if tblrNovo['B'] > tblrAntigo['B']:
        tblrAntigo['B'] = tblrNovo['B']

    if tblrNovo['R'] > tblrAntigo['R']:
        tblrAntigo['R'] = tblrNovo['R']

    return tblrAntigo

def flood_fill(img, x, y, height, width, tblr, first):

    if img[y][x] == 0:
        return

    label = tblr['label']

    if first:
        first = False
        tblr['T'] = y
        tblr['B'] = y
        tblr['L'] = x
        tblr['R'] = x
    else:
        if x < tblr['L']:
            tblr['L'] = x
        else:
            if x > tblr['R']:
                tblr['R'] = x

        if y < tblr['T']:
            tblr['T'] = y
        else:
            if y > tblr['B']:
                tblr['B'] = y

    img[y][x] = label

    #Topo
    if (y - 1) >= 0 and img[y-1][x] == -1:
        tblrNovo = flood_fill(img, x, y - 1, height, width, tblr, first)
        tblr = compara_tblr(tblr, tblrNovo)
    #Baixo
    if (y + 1) < height and img[y+1][x] == -1:
        tblrNovo = flood_fill(img, x, y + 1, height, width, tblr, first)
        tblr = compara_tblr(tblr, tblrNovo)
    #Esquerda
    if (x - 1) >= 0 and img[y][x-1] == -1:
        tblrNovo = flood_fill(img, x - 1, y, height, width, tblr, first)
        tblr = compara_tblr(tblr, tblrNovo)
    #Direita
    if (x + 1) < width and img[y][x+1] == -1:
        tblrNovo = flood_fill(img, x + 1, y, height, width, tblr, first)
        tblr = compara_tblr(tblr, tblrNovo)

    tblr['n_pixels'] += 1
    return tblr
def segmentacao_por_otsu(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def bordas_canny(img):
    bordas = cv2.Canny(img, 122, 123)
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return bordas, contornos

def main():
    sys.setrecursionlimit(100000)
    img = cv2.imread(INPUT_IMAGE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #hist, bins = np.histogram(gray, bins=256, range=[0, 256])

    normalizedImg = np.zeros((gray.shape[0], gray.shape[1]))
    normalizedImg = cv2.normalize(gray,  normalizedImg, 0, 255, cv2.NORM_MINMAX)


    #plt.plot(hist)
    #plt.show()

    #peaks, _ = find_peaks(hist, distance=50)
    #np.diff(peaks)
    #plt.plot(hist)
    #plt.plot(peaks, hist[peaks], "x")
    #plt.show()

    cv2.imshow('Imagem gray', gray)
    #cv2.imshow('Imagem Segmentadaa', normalizedImg)
    cv2.waitKey(0)
    blur = cv2.GaussianBlur(normalizedImg, (11, 11), 0)
    cv2.imshow('Imagem Segmentada', blur)
    cv2.waitKey(0)

    #img_segment = segmentacao_por_otsu(gray)
    img_segment = cv2.adaptiveThreshold(blur, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, -7)
    cv2.imshow('Imagem Segmentada', img_segment)
    cv2.waitKey(0)

    #img_segment = segmentacao_por_otsu(normalizedImg)
    componentes = rotula(img_segment, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    print(componentes)
    # Extraia os valores de 'n_pixels' para encontrar a mediana
    valores_n_pixels = [dicionario['n_pixels'] for dicionario in componentes]

    # Calcule a mediana dos valores
    mediana = int(statistics.median(valores_n_pixels))

    arroz = 0
    for i in range(0, len(valores_n_pixels)):
        resultado_divisao = float(valores_n_pixels[i] / mediana)
        # Arredondar para cima se a casa decimal for maior que 0.5
        if resultado_divisao - int(resultado_divisao) > 0.5:
            resultado_divisao_arredondado = math.ceil(resultado_divisao)
        else:
            resultado_divisao_arredondado = int(resultado_divisao)
        arroz+=resultado_divisao_arredondado

    edges = cv2.Canny(img_segment, 100, 145)

    # Imprima a mediana
    valores_n_pixels.sort()
    print("Ordenado:", valores_n_pixels)
    print("Mediana dos n_pixels:", mediana)
    print("Quantidade arroz", arroz)
    #Exiba a imagem segmentada
    #print(len(contornos))
    #cv2.imshow('Imagem com blurr', blur)
    cv2.imshow('Imagem Segmentada', img_segment)
    cv2.imshow('Imagem Bordas', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#    cv2.imwrite('03 - imagem_com_bloom.png', imagem_com_bloom)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()