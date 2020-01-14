
# coding: utf-8

# In[ ]:


#      Procedura 1
# testowanie na wygenerowanym zbiorze wyg.dane_aktywnosc
#Glowna procedura
#main()

# Notation           Indication
#indywiduum - wektor produktow na ktorych klient NIE dokonal aktywnosci
#iniPop      -The initial population. Inicjalizujaca populacja ((indywidua) produkty na ktorych klient nie dokonal aktywnosci)
#topX        -The percentage of individual that should be selected. (indywidua z najwiekszym wskaznikiem predykcji)
#bestMem     -The topX of the fitter individuals.  (najlepsze indywidua)
#CrossOverP  -The crossover probability. (wartosc parametur krzyzowki)
#mutP        -The mutation probability.  (wartosc parametru mutacji)
#recommendMem-The best individual that will be recommended its items to the active user. (rekomendacje - najlepsze indywiduum)
#maxGan      -The maximum number of generations. (maksymalna liczba generacji)
#currentGen  -The current generation.  (obecna generacja)
#nextGen     -The following generation.  (nastepna generacja)
#targetUser  -The active user.    (docelowy klient)
#M           -The size of the population.    (rozmiar populacji)
#N           -The number of items within the individual.   (liczba produktow w indywiduum)
#Z           -An individual.    (indywiduum (inaczej osobnik w algorytmie genetycznym))
#Selected    -The users who rated at least one item within an individual.


#


#1 Przygotowanie macierzy Klient-Produkt - wartosci macierzy =obliczenie wspolczynnika wg. wzoru (1) (zakup > dodanie do koszyka > wyswietlenie )
#2 Przygotowanie macierzy Produkt-Kategoria
#3 Wygenerowanie inicjalizujacej populacji ktora zawiera M indywiduów
#    Obliczamy wartosc wspolczynnika zainteresowania ze wzoru nr.1 
#4 For CurrentGen =/= maxGen
    #obliczamy wartosc korelacji (correlation(z)) kazdego indywiduum w iniPop.
    #Wybieramy najlepsze (topX) indywiduum(bestMem) na podstawie wartosci korelacji
    #Ustalamy operator krzyzowki z prawdopodobienstwem = CrossOverP
    #Ustaamy operator mutacji z prawdopodobienstwem = mutP
    #obliczamy wartosc podobienstwa (Similarity(z)) kazdego indywiduum w populacji inicjalizujacej (iniPop)
    #Wybieramy najlepsze (topX) indywidua (bestMem) na podstawie wartosci podobienstwa
#End For
#5 Obliczamy wskaznik prognozy(predykcji) (predict(z,x)) kazdego indywiduum w populacji iniPop
#6 Wybieramy najlepsze indywiduum(recommendMem) w iniPop


# In[ ]:


#    zakladam takie wartosci parametrow
#Parameter Value
#topX       25%
#CrossOverP 80%
#mutP       20%
#maxGen     100
#M          50
#N          10


# In[ ]:


#Procedura 2
#initalPop()

#1. Ustalamy produkty na ktorych klient dokonal aktywnosci
#2. Generujemy M indywiduow zawierajacych N produktow na ktorych klient nie dokonal aktywnosci
# indywiduum =  [Ia,Ib,Ic,Id,Ie,If,Ig,Ik] Ia,Ib,Ic,... - produkty


# In[ ]:


#Procedura 3
#Korelacja(z)

#1. Definiujemy kazdy produkt w z(indywiduum) poprzec wektor Vi = (vi1,vi2,...,vik) jezeli vi1 = 1 - produkt nalezy do kategorii 1, vi1=0 - nie nalezy
#2. For i1 =/= N-1        /// N - liczba produktow w indywiduum(parametr)                                 
        #For i2 =/=N
            #Obliczamy korelacje pomiedzy i1 a i2 wg wzoru 3.(metryka jaccard)  / 𝑐𝑜𝑟𝑟(𝑝, 𝑞) =𝐶11/𝐶10 + 𝐶01 + 𝐶11    ///p,q - produkty -from sklearn.metrics import jaccard_similarity_score
        #End For
    #End For
    #Obliczamy wartosc funkcji dopasowania z poprzez zsumowanie wartosci korelacji produktow wg wzoru 2.   /) 𝑐𝑜𝑟𝑟𝑒𝑙𝑎𝑡𝑖𝑜𝑛 𝑧 =E corr(p, q)(3)
    


# In[ ]:


#Procedura 4
#Podobienstwo(z)

# Ustalalmy wszystkich klientow ktorzy dokonali aktywnosci na przynajmniej jednym produkcie w z (selectedU)
# For size =/= size of selecetedU   
#    u = selectedU(size)
#    Obliczamy wartosc podobienstwa pomiedzy docelowym klientem (targetUser) a u wg wzoru.6 (korelacja pearsona)
#    Obliczamy wartosc podobienstwa Jaccarda pomiedzy docelowym klientem a u wg wzoru 7.
#    Normalizujemy wartosc podobienstwa pomiedzy docelowym klientem a u wg wzoru 5
# End For

#2.Obliczamy wartosc funkcji dopasowania z poprzez zsumowanie wartosci podobienstwa docelowago klienta
#a klinetami ktorzy dokonali aktywnosci na przyjajmniej jednym produkcie w z wg. wzoru 4.


# In[ ]:


#Procedura 5
#Predykcja(bestMem)
#1.For each z in bestMem
#     For each i in z
#        Prognozujemy wskaznik i dla docelowego klienta wg wzoru 9.
#     End For
#     Obliczamy wartosc funkcji dopasowania z poprzez zsumowanie wskaznika predykcji produktow wg wzoru 9.
#  End For
#2.Wybieramy najlepsze indywiduum(recommendMem) na podstawie wartosci funkcji dopasowania z bestMem


# In[41]:


#procedura 2 - initialPop()
import pandas as pd
import numpy as np

dane = pd.read_csv('C:\\Users\\mateusz\\Desktop\\wyg.dane_aktywnosc.csv')
ramkadanych = pd.DataFrame(np.zeros(shape=(80,264)),
                             columns=dane['produktid'].unique(),
                             index=dane['klientid'].unique())
ramkadanych.head(15)
dane = dane.groupby(['klientid', 'produktid'])['zdarzenie'].sum().reset_index()
danedf = pd.DataFrame(dane, columns = ['klientid','produktid','zdarzenie'],dtype = int)
Ramkad = danedf.pivot(index = 'klientid', columns ='produktid', values = 'zdarzenie').fillna(0)
Ramkad.head(15)

def iniPopWys(rd,idklienta):
    targetUser = rd.loc[:,rd.loc[idklienta] != 0].columns
    return targetUser
iniPopWys(Ramkad,12)

def initialPop(rd,idklienta):
    targetUser = rd.loc[:,rd.loc[idklienta] == 0].columns
    
    return targetUser

ram = initialPop(Ramkad,12)
ls1 = list(ram)
ls1 # wszystkie produkty na ktorych klient nie dokonal aktywnosci , zapisane w liscie
#teraz nalezy stworzyc N indywiduow , (dzielac liste na N(przyjalem =10) list)
x = range(20) 
l = np.array_split(np.array(ls1),20)  #jakos tak
l
l[1]


# In[122]:


#procedura 3 - korelacjaZ()
#obliczanie korelacji miedzy produktami
#zapisujemy kazdy produkt w indywiduum jako wektor() np. v1 - (1,0,1,...) - produkt 1 nalezy do kategori 1,3 itd.
#obliczamy korelacje miedzy produktami np. i1 i i2  - uzywajac metryki binarnej Jaccarda - from sklearn.metrics import jaccard_similarity_score
#obliczamy wartosc funkcji dopasowania sumujac wartosc korelacji produktow
from scipy.stats import pearsonr
import seaborn
kategorie = pd.read_csv('C:\\Users\\mateusz\\Desktop\\wyg.dane_kategorie.csv')
kategorie.head(15)
katvec = kategorie.values
katvec[1]
k1 = np.delete(katvec[1],0)
k2 = np.delete(katvec[2],0)
k1.astype(float)
k2.astype(float)
np.corrcoef(k1.astype(float),k2.astype(float))


# In[97]:


#procedura 3
#ustalamy wszystkich klientow ktorzy dokonali aktywnosci na przynajmniej jednym produkcie w Z(wybrane indywiduum)

