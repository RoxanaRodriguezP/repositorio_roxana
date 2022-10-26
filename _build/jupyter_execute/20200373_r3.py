#!/usr/bin/env python
# coding: utf-8

# # REPORTE 3

# ## Código 
# Greysi Arrelucea 20200279 - Roxana Rodriguez Pilco 20200373

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import pandas as pd
import numpy as np
import random
import math
import sklearn
import scipy as sp
import networkx
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from causalgraphicalmodels import CausalGraphicalModel


# ### 1. Explique cuáles son los intrumentos de política monetaria que puede utilizar el Banco Central.
# 

# El BCRP es la autoridad monetaria central del sistema bancario. Este tiene la capacidad de gestionar la oferta monetaria al aplicar políticas monetarias expansivas y contractivas por medio de 3 posibles instrumentos: Operación de mercado abierto, el coeficiente legal de encaje y la tasa de interés.
# 
# En primer lugar, el Banco Central usa como instrumento a la Operación de Mercado Abierto para aumentar o disminuir la oferta monetaria. Con operaciones de mercado abierto se hace referencia a las operaciones de compra y venta de activos financieros o bonos a los bancos comerciales. Por medio de este instrumento el BCRP realiza políticas expansivas cuando compra bonos del mercado, pues inyecta dinero (liquidez) en la economía, lo que contribuirá al aumento de la oferta monetaria. También realiza políticas contractivas por medio de la venta de bonos al mercado, pues al hacerlo retira dinero de la economía, lo que deriva en la reducción de la oferta monetaria.
# 
# En segundo lugar, el BCRP usa el coeficiente legal de encaje para realizar políticas expansivas y contractivas y controlar la oferta monetaria: Por un lado, al reducir la tasa de encaje permite que los bancos dispongan de más dinero para realizar préstamos porque aumenta el multiplicador bancario y este aumento del dinero bancario significa un aumento de la oferta monetaria (política expansiva). Por otro lado, el BCRP puede aumentar el coeficiente legal de encaje, haciendo que los bancos comerciales deban tener una mayor proporción de depósitos en forma de reservas, lo que significaría la reducción del
# multiplicador bancario y en consecuencia una disminución de la oferta monetaria (política contractiva).
# 
# En tercer lugar, el Banco Central viene utilizando la tasa de interés como una herramienta para hacer políticas expansivas y contractivas en el mercado monetario desde 1990, cuando cambió su esquema de política monetaria. Para este caso, se comprende a la oferta monetaria como una variable endógena a diferencia de los 2 casos anteriores. De modo que, por un lado, el BCRP hace política monetaria expansiva al reducir su tasa de interés de referencia, lo que aumenta la cantidad de dinero prestada a los bancos comerciales y la base monetaria, generando con ello el aumento de la oferta monetaria. Por otro lado, este hace política monetaria contractiva cuando aumenta su tasa de interés de referencia, pues con ello disminuye la cantidad de dinero prestada a los bancos comerciales al igual que la base monetaria, generando en consecuencia la disminución de la oferta monetaria.

# ### 2. Derive la oferta real de dinero y explique cada uno de sus componentes.
# 
# $$ \frac{M^s}{P}=\frac{M^s_0}{P_0}$$
# 
# - Explicación:
# 
# 
# Esta fórmula representa la cantidad de dinero que circula en la economía ($M^s$). En esta fórmula se supone que la oferta nominal de dinero es una variable exógena y un intrumento de política monetaria. Para expresar en términos reales, es decir, mostrando lo que verdaderamente vale el dinero, se divide la oferta nominal de dinero entre el nivel general de precios de la economía ($P$). Además, para un análisis de corto plazo el nivel de precios es constante y está determinado.

# ### 3. Derive la demanda real de dinero. Explique qué papel cumplen los parametros "k" y "j"
# 
# La demanda de dinero sucede por distintos motivos, los cuales dan forma a la ecuación de la demanda real de dinero.
# Entre estos motivos se encuentran los factores de transacción y precaución. El primero hace referencia a que la demanda de dinero sucede por su función de medio de intercambio, pues este sirve para realizar transacciones. De modo que, la magnitud de estas transacciones depende directamente del nivel de Ingreso ($Y$). El segundo motivo se refiere a que se demanda dinero como una precaución, pues de esta manera se tendría dinero para amortiguar en el futuro posibles deudas, préstamos y las cuotas de intereses en plazos fijos que estas deudas conllevan. Además, ya que la capacidad de pago de las personas depende de sus ingresos, podemos decir que la demanda de dinero por motivos de precaución también depende positivamente del nivel de ingreso($Y$). La demanda de dinero por los motivos transacción y precaución, da forma a la ecuación: 
# 
# $$L_1=kY$$
# 
# El tercer motivo es el de especulación. Este está relacionado con el sector financiero y establece que los actores preferirán mantener liquidez en forma de dinero y no en forma de bonos cuando la tasa de interés de los bonos se reduce, por ende, la demanda de dinero dependería inversamente de la tasa de interés de los bonos(deuda). La demanda de dinero por motivo de especulación se representa así: 
# 
# $$L_2= -ji$$
# 
# - Derivación
# 
# Asumiendo que tanto la demanda por motivos de transacción y precaución ($L_1$) como la demanda por motivo de especulación ($L_2$) están en términos reales, se obtiene que la demanda de dinero real es:
# 
# $$M_d=L_1+L_2$$
# 
# Siendo $L_1$ y $L_2$: 
# 
# $$L_1=kY$$ 
#     
# donde $k$ representa la elasticidad o sensibilidad de la demanda de dinero ante variaciones del nivel de ingreso $Y$.
# 
# $$L_2=-ji$$
# 
# donde $j$ representa la sensibilidad de la demanda de dinero ante variaciones de la tasa de interés nominal de los bonos $i$
# 
# Entonces, reemplazando valores:
# 
# $$M_d=kY-ji$$
# 
# Como asumimos que la tasa de interés nominal es igual a la real $i=r$, entonces la función de demanda real de dinero es:
# 
# $$M_d=kY-jr$$
# 
# - Explicación del papel de $k$ y $j$
# 
# A partir de la función de la demanda de dinero ($M^d$), se obtienen los parámetros $k$ y $j$, ambos con diferentes roles. Por un lado, el parámetro $k$ se encarga de indicar la sensibilidad de la demanda de dinero ante variaciones del ingreso $Y$. Mientras más grande sea el valor de $k$, mayor cantidad de dinero se demandará en cuanto se incremente el nivel de ingreso de la economía. Por otro lado, el parámetro $j$ va a señalar cuan sensible es la demanda de dinero($M^d$) ante variaciones de la tasa de interés nominal de los bonos ($i$), pero si asumimos que la tasa de interés nominal es igual a la real ($i=r$), entonces, indicará que tan sensible es la demanda de dinero ante las variaciones de la tasa de interés real ($r$). Se demanda más cantidad de dinero cuando la tasa de interés baja y la demanda de dinero será menor cuando la tasa de interés sube.
# 
# 

# ### 4. Asumiendo que no hay inflación podemos asumir que $i=r$. Escriba en terminos reales la ecuación de equilibrio en el mercado de dinero.
# 
# El equilibrio en el Mercado de Dinero se deriva del equilibrio entre la oferta de dinero $M^s$ y la demanda de dinero $M^d$, resultando:
# 
# $$M^s=M^d$$
# 
# Reemplazando las ecuaciones ya mostradas en los apartados anteriores se obtiene:
# 
# $$ \frac{M^s}{P}=kY-ji$$
# 
# Si se asume que la inflación esperada es cero, ya que a corto plazo el nivel de precios es fijo y exógeno. Entonces, no habría una gran diferencia entre la tasa de interés nominal y la real y podemos decir que $i=r$.
# 
# Siendo así, la ecuación de equilibrio en el mercado monetario en términos reales es:
# 
# $$ \frac{M^s}{P}=kY-jr$$
# 
# $$ M^s=P(kY-jr)$$

# ### 5. Grafique el equilibrio en el mercado de dinero.

# In[2]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 36
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[3]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Equilibrio en el mercado de dinero", xlabel='$M^s / P$', ylabel='r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#346beb')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "#34eb9c")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=8, xmin= 0, xmax= 0.5, linestyle = ":", color = "#ada8a8")

# Agregamos texto
ax1.text(0, 8.5, "$r^e_0$", fontsize = 12, color = 'black')
ax1.text(83, 2, "$L(Y_0)$", fontsize = 12, color = 'black')
ax1.text(51, 4, "$M^s_0/ P_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ### 6. Estática comparativa en el Mercado de Dinero

# #### 6.1. Explique y grafique qué sucede en el mercado de dinero si $ΔY<0$

# En términos simples, cuando a la economía no le va tan bien, el nivel de producción del país ($Y$) se reduce. Las personas se enteran de esto y prefieren tener menos liquidez en forma de dinero por diferentes motivos y como el BCRP no ha cambiado la cantidad de dinero, es decir, la oferta de dinero se mantiene fija. Entonces, como no hay variación en la cantidad de dinero ofertada y los pobladores demandan cada vez menos dinero (la cantidad de dinero demandada es menor que la cantidad de de dinero ofertada, pues esta se mantiene fija)la tasa de interés baja.
# 
# Por tanto, cuando se reduce el ingreso $Y$, la demanda de dinero ($M^d$) también se reduce. Entonces, para el mercado que estaba en equilibrio, esta disminución del ingreso genera una reducción de la demanda ($M^d<M^s$) en el mercado, dando como resultado una reducción de la tasa de interés ($r$), la cual pasa de $r^e_0$ a $r^e_1$. Por tanto, la curva de demanda de dinero se desplaza hacia la izquierda (hacia abajo).

# In[4]:


# Parametros
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 36
MS_0 = 500

r = np.arange(r_size)

# Creamos la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Creamos la oferta de dinero.
MS = MS_0 / P
MS


# In[5]:


#--------------------------------------------------
    # NUEVA curva de equilibrio

# Definir parámetros con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y_1 = 20
MS_0 = 500

r = np.arange(r_size)

# Creamos la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)

# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo al nombre de las coordenadas
ax1.set(title="Equilibrio en el mercado de dinero", xlabel='$M^s / P$', ylabel='r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#800080')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "blue")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=0, xmin= 0, xmax= 0.5, linestyle = ":", color = "grey")

# Agregamos texto para la curva
ax1.text(0, 8.5, "$r^e_0$", fontsize = 12, color = 'black')
ax1.text(50, -5, "$M^s_0/P_0$", fontsize = 12, color = 'black')
ax1.text(50, 8, "$E_0$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#20b2aa')
ax1.text(0, 0.5, "$r^e_1$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$E_1$", fontsize = 12, color = 'black')

# Creamos las lineas puntadas para el nuevo equilibrio
ax1.axhline(y=8, xmin= 0, xmax= 0.5, linestyle = ":", color = "grey")

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# #### 6.2. Explique y grafique qué sucede en el mercado de dinero si $Δk<0$

# La demanda de dinero va a aumentar conforme aumente el parámetro $k$, por tanto, ante una reducción de este parámetro, la demanda de dinero disminuirá, lo que genera un desplazamiento hacia la izquierda(hacia abajo) de la curva de equilibrio del mercado de dinero. Esta reducción de la demanda de dinero generaría un exceso de oferta, pues la cantidad de dinero ofertado se mantiene fija, por ende la tasa de interés de equilibrio se reduce de $r^e_0$ a $r^e_1$.

# In[6]:


# Parámetros
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 36
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[7]:


# Parámetros con cambio en el nivel de k
r_size = 100

k_1 = 0.4
j = 0.2                
P  = 10 
Y = 36
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k_1, j, P, r, Y)

# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Equilibrio en el mercado monetario", xlabel= '$M^s / P$', ylabel='r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#800080')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "C6")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=4.3, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto para la curva
ax1.text(0, 8.5, "$r^e_0$", fontsize = 12, color = 'black')
ax1.text(50, -5, "$M^s_0/P_0$", fontsize = 12, color = 'black')
ax1.text(50, 8, "$E_0$", fontsize = 12, color = 'black')

# Nueva curva a partir del cambio en k
ax1.plot(MD_1, label= '$L_1$', color = '#20b2aa')
ax1.text(0, 4.8, "$r^e_1$", fontsize = 12, color = 'black')
ax1.text(50, 4.5, "$E_1$", fontsize = 12, color = 'black')

# Creamos las lineas puntadas para el nuevo equilibrio
ax1.axhline(y=8, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# #### 6.3. Explique y grafique qué sucede en el mercado de dinero si $ΔM^s<0$

# Cuando disminuye la oferta monetaria ($ΔM^s<0$), se genera una reducción de la oferta monetaria en el mercado, lo que en consecuencia da lugar a un aumento de la tasa de interés para que así aumente la demanda de dinero ($M^d$) y se restablezca el equilibrio en el mercado. La reducción de la cantidad de dinero va de ($M^s_0$) a ($M^s_1$), por tanto la recta de la oferta real de dinero se desplaza hacia la izquierda, aumentando la tasa de interés de equilibrio de $r^e_0$ a $r^e_1$.
# 
# En términos simples, cuando el BCRP disminuye la cantidad de dinero en la economía, la oferta agregada de dinero se contrae. Sin embargo, la demanda de dinero no se modificó, por tanto, se genera una situación donde el nivel de producción es demasiado a comparación de la demanda, pues los actores no quieren consumir tanto, por tanto el precio del dinero($r$)cae. De modo que, para que se ajuste el equilibrio, se debe ajustar el nivel de precios, lo que significa que la tasa de interés (precio del dinero) tiene que subir ($r↑$).

# In[8]:


# Parámetros
r_size = 100

k = 0.5
j = 0.2                
P_1  = 20 
Y = 36
MS_0 = 500

r = np.arange(r_size)

# Creamos la función de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P_1, r, Y)
# Creamos la oferta de dinero.
MS_1 = MS_0 / P_1
MS


# In[9]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Equilibrio en el mercado monetario", xlabel='$M^s / P$', ylabel='r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = 'C4')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "C2")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=8, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto para la curva Ms
ax1.text(0, 8.5, "$r^e_0$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$(M^s_0/P)_0$", fontsize = 12, color = 'black')
ax1.text(50, 8, "$E_0$", fontsize = 12, color = 'black')

# Nuevas curvas y textos a partir del cambio en P
ax1.axvline(x = MS_1,  ymin= 0, ymax= 1, color = "C2")
ax1.text(0, 13.5, "$r^e_1$", fontsize = 12, color = 'black')
ax1.text(25, 0, "$(M^s_1/P)_1$", fontsize = 12, color = 'black')
ax1.text(25, 13, "$E_1$", fontsize = 12, color = 'black')

# Creamos las lineas puntadas para el nuevo equilibrio
ax1.axhline(y=13, xmin= 0, xmax= 0.275, linestyle = ":", color = "black")

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ### 7. Curva LM

# #### 7.1. Derive paso a paso la curva LM matemáticamente (a partir del equilibrio en el Mercado Monetario) y grafique.

# 
# $$(M^s)=(M^d)$$
# 
# Siendo $(M^s)=\frac{M^s_0}{P_0}$ y $(M^d)=kY-jr$
# 
# Entonces, en términos nominales la ecuación de equilibrio sería igual a:
# 
# $$ \frac{M^s_0}{P_0}=(kY-ji)$$
# 
# Ya que, se puede suponer que la inflación esperada es cero, pues a corto plazo el nivel de precios es fijo y exógeno. Por lo tanto, no habría una gran diferencia entre la tasa de interés nominal ($i$) y real ($r$).
# 
# Siendo la ecuación del equilibrio en el mercado monetario:
# 
# $$ \frac{M^s_0}{P_0}=(kY-jr)$$
# 
# Se realizan ciertas operaciones algebraicas:
# 
# $$ kY-\frac{M^s_0}{P_0}=jr $$
# 
# $$ \frac{kY}{j} - \frac{M^s_0}{P_0j}=r $$
# 
# Entonces, la curva LM se da en función de la tasa de interés $r$:
#  
# $$ r = - \frac{1}{j}\frac{M^s_0}{P_0}  + \frac{k}{j}Y $$
# 
# Siendo, -$\frac{1}{j}\frac{M^s_0}{P_0}$ el intercepto y $\frac{k}{j} $ la pendiente

# In[10]:


#1----------------------Equilibrio mercado monetario

    # Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 36

r = np.arange(r_size)


    # Ecuación
def Ms_MD(k, j, P, r, Y):
    Ms_MD = P*(k*Y - j*r)
    return Ms_MD

Ms_MD = Ms_MD(k, j, P, r, Y)


    # Nuevos valores de Y
Y1 = 45

def Ms_MD_Y1(k, j, P, r, Y1):
    Ms_MD = P*(k*Y1 - j*r)
    return Ms_MD

Ms_Y1 = Ms_MD_Y1(k, j, P, r, Y1)


Y2 = 25

def Ms_MD_Y2(k, j, P, r, Y2):
    Ms_MD = P*(k*Y2 - j*r)
    return Ms_MD
Ms_Y2 = Ms_MD_Y2(k, j, P, r, Y2)

#2----------------------Curva LM

    # Parameters
Y_size = 100

k = 0.5
j = 0.2                
P  = 10               
Ms = 30            

Y = np.arange(Y_size)


# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[11]:


# Gráfico de la derivación de la curva LM a partir del equilibrio en el mercado monetario

    # Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8)) 


#---------------------------------
    # Gráfico 1: Equilibrio en el mercado de dinero
    
ax1.set(title="Equilibrio en el mercado de dinero", xlabel='$M^s / P$', ylabel='r')
ax1.plot(Y, Ms_MD, label= '$L_0$', color = '#41bf63')
ax1.plot(Y, Ms_Y1, label= '$L_1$', color = '#41bf63')
ax1.plot(Y, Ms_Y2, label= '$L_2$', color = '#41bf63')
ax1.axvline(x = 45,  ymin= 0, ymax= 1, color = "grey")

ax1.axhline(y=35, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=135, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=88.5, xmin= 0, xmax= 1, linestyle = ":", color = "black")

ax1.text(47, 139, "C", fontsize = 12, color = 'black')
ax1.text(47, 92, "B", fontsize = 12, color = 'black')
ax1.text(47, 39, "A", fontsize = 12, color = 'black')

ax1.text(0, 139, "$r_2$", fontsize = 12, color = 'black')
ax1.text(0, 92, "$r_1$", fontsize = 12, color = 'black')
ax1.text(0, 39, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()
 
#---------------------------------
# Gráfico 2: Curva LM
    
ax2.set(title="Curva LM", xlabel='Y', ylabel='r')
ax2.plot(Y, i, label="LM", color = '#8aa8e3')

#Lineas punteadas
ax2.axhline(y=160, xmin= 0, xmax= 0.69, linestyle = ":", color = "black")
ax2.axhline(y=118, xmin= 0, xmax= 0.53, linestyle = ":", color = "black")
ax2.axhline(y=76, xmin= 0, xmax= 0.38, linestyle = ":", color = "black")
ax2.axvline(x=70,  ymin= 0, ymax= 0.69, linestyle = ":", color = "black")
ax2.axvline(x=53,  ymin= 0, ymax= 0.53, linestyle = ":", color = "black")
ax2.axvline(x=36,  ymin= 0, ymax= 0.38, linestyle = ":", color = "black")

#Se colocan las letras
ax2.text(67, 164, "C", fontsize = 12, color = 'black')
ax2.text(51, 125, "B", fontsize = 12, color = 'black')
ax2.text(35, 80, "A", fontsize = 12, color = 'black')
#Se coloca texto para las curvas
ax2.text(0, 164, "$r_2$", fontsize = 12, color = 'black')
ax2.text(0, 125, "$r_1$", fontsize = 12, color = 'black')
ax2.text(0, 80, "$r_0$", fontsize = 12, color = 'black')
ax2.text(72.5, -14, "$Y_2$", fontsize = 12, color = 'black')
ax2.text(56, -14, "$Y_1$", fontsize = 12, color = 'black')
ax2.text(39, -14, "$Y_0$", fontsize = 12, color = 'black')

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.legend()

plt.show()


# #### 6.2. ¿Cuál es el efecto de una disminución en la Masa Monetaria ($M^s<0$)? Explica usando la intuición y gráficos.

# 
# $$ M^s_0↓ → \frac{M^s_0}{P_0} ↓ → r↓ → M^s↓ →M^s<M^d → r↑$$
# 
# Dado que se reduce la masa monetaria, esto provoca el aumento de la tasa de interés, pues para un mismo nivel de ingreso, cuando disminuye la oferta de dinero, la tasa de interés debe subir para que aumente la demanda de dinero y se restaure el equilibrio en el mercado monetario. El desplazamiento es hacia la izquierda

# In[12]:


#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 600             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
Ms = 75

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[13]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = '#784ea3')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#ff69b4', linestyle = 'dashed')

# Texto agregado
plt.text(47, 81, '∆$M^s$', fontsize=12, color='black')
plt.text(47.5, 73, '←', fontsize=15, color='black')

# Título y leyenda
ax.set(title = "Efectos de la disminución en la Masa Monetaria $(M^s)$", xlabel='Y', ylabel='r')
ax.legend()


plt.show()


# #### 6.3. ¿Cuál es el efecto de un aumento en k (k>0)? Explica usando intuición y gráficos.

# 
# $$ G_0↑ → DA↑ → DA>Y → r↑$$
# 
# Si observamos la curva LM, podemos ver que ante un aumento de $k$, la curva LM es más rígida, desplazándose hacia la izquierda, provocando que, si la renta aumenta, sea necesario que el tipo de interés ($r$) también aumente para restablecer el equilibrio monetario $E$. 
# 

# In[14]:


#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 600             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
k = 5

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[15]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = '#1f66e5')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3ab4e0', linestyle = 'dashed')

# Texto agregado
plt.text(63, 185, '∆$k$', fontsize=12, color='black')
plt.text(63, 170, '←', fontsize=15, color='black')

# Título y leyenda
ax.set(title = "Efectos del aumento en la elasticidad del ingreso $(k)$", xlabel='Y', ylabel='r')
ax.legend()


plt.show()


# ## Lectura

# La Macroeconomía de la cuarentena: Un modelo de dos sectores es una investigación escrita por Waldo Mendoza, Luis Mancilla y Rafael Velarde. Esta investigación tiene como objetivo presentar un modelo macroeconómico inspirado en Blanchard (2021) que replique los hechos acontecidos durante 2020-2021 – la expansión del COVID-19 y la cuarentena como medida de contención– y sus efectos en la economía peruana. De modo que, a través de este modelo se simulan los efectos macroeconómicos de la cuarentena –en el nivel de producción y el nivel de precios– de manera indirecta o directa en una economía como la peruana que cuenta con 2 sectores. El sector 1 está compuesto por hoteles, restaurantes, líneas aéreas, entre otros, los cuales fueron afectados de manera directa por la orden de cese de actividades durante la cuarentena, lo que generó la disminución del PBI potencial y la disminución del consumo autónomo en este sector. El sector 2 abarca los sectores de producción de alimentos, y bienes y servicios indispensables. Este sector si bien siguió operando durante la cuarentena, fue afectado indirectamente por el shock de demanda negativo del sector anterior. 
# 
# Es así que, la cuarentena y sus efectos restrictivos para los sectores genera la contracción del PBI en el sector 1 y consecuentemente, la disminución de los ingresos de los trabajadores de este sector afectó la demanda y la producción en el sector 2 e igualmente la reducción de ingresos en el sector 2 afecta la demanda y PBI del sector 1. Así pues, dado el temor por la pandemia, el consumo se reduce y se produce un choque generalizado sobre la demanda que deriva en la contracción de la producción para ambos sectores. En ese marco, la pregunta de investigación sería: ¿De qué manera se puede simular los efectos macroeconómicos de una cuarentena en una economía de 2 sectores? 
# La respuesta a la interrogante será respondida con el planteamiento de un modelo inspirado en el modelo IS-MR-DA-OA que propone Blanchard en 2021, el cual simulará los efectos de la aplicación de la cuarentena a corto plazo, en el equilibrio estacionario y en el tránsito hacia el equilibrio estacionario y también mostrará los efectos del fin de su vigencia.
# 
# Asimismo, el enfoque utilizado por los autores es cuantitativo, pues presenta la creación de un modelo que evaluará los efectos de la cuarentena en el caso de una economía de 2 sectores como la peruana. Este modelo presenta ciertas características: el sector 1 y 2 son keynesianos, ya que el nivel de producción es determinado por la demanda y el nivel de precios dependerá del nivel esperado de precios y de la brecha del PBI en cada sector y hay una conexión entre sectores producida por el consumo de los trabajadores en ambos sectores. En conjunto, el enfoque que sigue la investigación y el detalle de su presentación permite profundizar en sus fortalezas y debilidades.
# 
# Es así que una fortaleza del enfoque es que la elaboración de un modelo macroeconómico va a ser un simulador de la realidad, característica que le añade veracidad a la investigación, ya que replicará de manera dinámica un fenómeno macroeconómico como es la cuarentena y sus efectos en un caso como es la economía peruana de 2 sectores. De forma que, este modelo se convertirá en una base de información y predicción de comportamientos macroeconómicos que será tomado como una herramienta útil para futuras investigaciones. Igualmente, otra fortaleza del enfoque es que permite el uso de gráficos estadísticos para mostrar los acontecimientos sucedidos en el periodo 2020-2021 en la economía peruana, lo que generará que la presentación de la información sea más dinámica e integral y que los lectores por si mismos comprueben la veracidad de la información presentada en el estudio por medio de los gráficos y a su vez la funcionalidad del modelo. Sin embargo, una debilidad inevitable del enfoque es que al presentar un modelo tendrá que tomar en cuenta que, si bien es funcional bajo ciertos aspectos de la economía de 2 sectores, habrá aspectos que no tomará en cuenta, ya que simula la realidad y los efectos macroeconómicos de un fenómeno de manera reducida. Por ende, el modelo no abarcará algunas variables. En este sentido, existirá un margen de error que puede servir para cuestionar su significancia.
# 
# En suma, esta investigación tiene una importante contribución para la respuesta a la pregunta de investigación, pues por medio del modelo presentado se puede tener mayor detalle sobre los sectores de la economía peruana y los efectos macroeconómicos que tuvo el periodo de cuarentena 2020-2021 en el desarrollo de la economía peruana. Además, gracias al modelo se puede tener conocimiento del comportamiento de variables macroeconómicas como el caso del crédito bancario, quien tuvo un papel central en la recuperación económica en el Perú. Asimismo, la investigación permite evaluar el funcionamiento y papel resaltante del proyecto Reactiva Perú como parte del programa crediticio implementado en conjunto por el MEF y el BCRP para lograr la recuperación de las MYPES, empresas gravemente afectadas por el confinamiento y la estabilidad económica. Así también, contribuye a la mejora de las políticas macroeconómicas ante un eventual choque externo que amenace con poner a la economía peruana en crisis, pues este modelo y sobre todo el análisis del caso peruano sirve como una guía y evita que las autoridades puedan caer en un ciclo interminable de errores y aciertos. 
# 
# En conclusión, ahora que ya se respondió de cierta manera a la pregunta por medio del modelo, se presentó los efectos macroeconómicos del confinamiento y sus efectos al finalizar su aplicación en la economía y también maneras de solucionar la contracción económica por medio de programas crediticios como Reactiva Perú es momento de probar nuevos enfoques y variables para continuar con la respuesta a la pregunta. Por ejemplo, tal como menciona Mendoza (2018) un paso a seguir sería proponer que el modelo tome en cuenta que existen agentes que basan su comportamiento en expectativas racionales, pues con el propósito de introducir dinámica en el modelo presentado, los autores asumieron que en el subsistema de tránsito al equilibrio estacionario “los agentes económicos proyectan y crean sus expectativas solo a partir de la información de los precios en el periodo previo”, pero si se sigue la propuesta de implementar agentes económicos con expectativas racionales, estos no solo formarían sus expectativas a partir de la información de precios de los periodos previos sino que usarían toda la información relevante previa y actual, lo que supondría incluso que se anticipen a las medidas del Estado. De este modo, el precio esperado que se tuvo en cuenta para las ecuaciones de oferta agregada de ambos sectores podría igualarse al precio de equilibrio, lo que podría cambiar la dinámica de tránsito al equilibrio estacionario en el modelo establecido. 
# 
# Otro paso a seguir podría ser realizar otro modelo o una nueva investigación que no esté enfocada solo en los componentes de demanda para plantear políticas macroeconómicas de estabilización sino que se enfoque en otras dimensiones, pues estos efectos macroeconómicos de la cuarentena pueden haber sido intensificados por otras variables como el tiempo de duración del confinamiento o como mencionan Barrutia et al. (2021) puede ser que la economía ya haya estado sufriendo retrocesos que fueron intensificados por las medidas de confinamiento, por lo que, crear un modelo que también tome en cuenta otras dimensiones sería relevante para avanzar con la respuesta a la pregunta.
# 
# - Referencias:
# 
# Mendoza, W. (2018). Macroeconomía intermedia para América Latina (3ra ed.). Fondo Editorial PUCP.
# 
# 
# Barrutia Barreto, I., Silva Marchan, H. A., & Sánchez Sánchez, R. M. (2021). Consecuencias económicas y sociales de la inamovilidad humana bajo COVID-19: caso de estudio Perú. Lecturas de economía, (94), 285-303.
# 
# 

# In[ ]:




