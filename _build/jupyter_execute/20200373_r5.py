#!/usr/bin/env python
# coding: utf-8

# # REPORTE 5

# ## Código
# Greysi Arrelucea (20200279) y Roxana Rodriguez Pilco (20200373)

# In[1]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
#from causalgraphicalmodels import CausalGraphicalModel


# #### A partir del siguiente sistema de ecuaciones que representan el modelo DA-OA
# 

# ### 1. Encuentre las ecuaciones de Ingreso $Y^e$ y tasa de interés  de equilibrio $r^e$ (Escriba paso a paso la derivación de estas ecuaciones).

# Considernado, por un lado, la Curva IS:
# 
# $$ r = \frac{B_o}{h} - \frac{B_1}{h}Y $$
# 
# Donde $ B_0 = C_o + I_o + G_o + X_o $ y $ B_1 = 1 - (b - m)(1 - t)$

# Y, por otro, la ecuación de la LM
# 
# $$  r = -\frac{1}{j}\frac{Mo^s}{P} + \frac{k}{j}Y $$

# Eliminando “r” y despejando P, se obtiene:
# 
# $$  P = -\frac{h Mo^s}{-j B_o + (jB_1 + hk)Y} $$
# 
# Y, en función del nivel de ingresos $(Y)$ se obtiene:
# 
# $$  Y = \frac{jB_o}{jB_1 + hk} + (\frac{hMo^s}{jB_1 + hk})\frac{1}{P} $$

# Ahora bien, considerando la ecuación de equilibrio en el mercado monetario
# 
# $$ Mo^s - P = kY - jr $$
# 
# Se reemplaza $(r)$, y se obtiene la ecuación de la demanda agregada $(DA)$
# 
# $$  P = \frac{h Mo^s + jB_o}{h} - \frac{jB_1 + hk}{h}Y $$
# 

# - Oferta Agregada en el corto plazo:
# 
# El corto plazo es un periodo en el cual el producto $(Y)$ se ubica por debajo o por encima de su nivel de largo plazo o Producto Potencial $(\bar{Y})$.
# 
# Entonces, la curva de $OA$ de corto plazo se puede representar con la siguiente ecuación:
# 
# $$ P = P^e + θ(Y - \bar{Y}) $$ 
# 
# - Donde $(P)$ es el nivel de precios, $(P^e)$ el precio esperado y $\bar{Y}$ el producto potencial.

# - Equilibrio:
# 
# Para hallar los valores de equilibrio de las variables $r$,$Y$ y $P$ se empieza por igualar las ecuaciones de demanda agregada $DA$ y oferta agregada $OA$.
# 
# Considerando la ecuación de la demanda agregada $(DA)$:
# 
# $$  P = \frac{h Mo^s + jB_o}{h} - \frac{jB_1 + hk}{h}Y $$
# 
# Y la ecuación de la oferta agregada $(OA)$:
# 
# $$ P = P^e + θ(Y - \bar{Y}) $$  
# 

# Para hallar el nivel de producción de equilibrio ($Y^e$) igualamos ambas ecuaciones:
# 
# $$ \frac{h Mo^s + jB_o}{h} - \frac{jB_1 + hk}{h}Y = P^e + θ(Y - \bar{Y}) $$
# 
# $$ Y^e = [\frac{1}{(θ + \frac{jB_1+hk}{h})}] * [\frac{(hM^s_o + jB_o}{h}-P^e+θ\bar{Y})]$$

# - Para hallar $P^e$, despejamos la ecuación de $OA$ en función de Y, es decir, reemplazamos $Y^e$ en la ecuación de oferta agregada $OA$:
# 
# $$ P^{eq-da-oa}= P^e + θ(Y^{eq-da-oa} - \bar{Y}) $$ 
# 
# Y reemplazamos $Y$ en la ecuación de $DA$:
# 
# $$ P^{eq-da-oa} = P^e+θ([\frac{1}{(θ+\frac{jB_1+hk}{h})}] * [(\frac{hM^s_o + jB_o}{h}- P^e + θ\bar{Y})]- \bar{Y})$$

# Para encontrar la tasa de interés de equilibrio $r^{eq-da-oa}$ reemplazamos $P^{eq-da-oa}$ en la ecuación de tasa de interés de equilibrio del modelo IS-LM.
# 
# La tasa de interés de equilibrio del modelo IS-LM es:
# 
# $$r^{eq-is-lm}=\frac{kB_o}{kh+jB_1}-(\frac{B_1}{kh+jB_1})*(M^s_o-P)$$
# 
# Reemplazando, la tasa de interés de equilibrio en DA-OA es:
# 
# $$r^{eq-is-lm}=\frac{kB_o}{kh+jB_1}-(\frac{B_1}{kh+jB_1})*(M^s_o-P^{eq-da-oa})$$
# 
# $$r^{eq-da-oa}=\frac{kB_o}{kh+jB_1}-(\frac{B_1}{kh+jB_1})*(M^s_o-P^e+θ([\frac{1}{(θ+\frac{jB_1+hk}{h})}] * [(\frac{hM^s_o + jB_o}{h}- P^e + θ\bar{Y})]- \bar{Y}))$$
# 

# Por tanto, los valores de equilibrio de estas 3 variables $Y^{eq-da-oa}$, $r^{eq-da-oa}$ y $P^{eq-da-oa}$ son:
# 
# $$ Y^ {eq-da-oa}= [\frac{1}{(θ + \frac{jB_1+hk}{h})}] * [\frac{(hM^s_o + jB_o}{h}-P^e+θ\bar{Y})]$$
# 
# 
# $$ r^{eq-da-oa}=\frac{kB_o}{kh+jB_1}-(\frac{B_1}{kh+jB_1})*(M^s_o-P^e+θ([\frac{1}{(θ+\frac{jB_1+hk}{h})}] * [(\frac{hM^s_o + jB_o}{h}- P^e + θ\bar{Y})]- \bar{Y}))$$
# 
# 
# $$ P^{eq-da-oa} = P^e+θ([\frac{1}{(θ+\frac{jB_1+hk}{h})}] * [(\frac{hM^s_o + jB_o}{h}- P^e + θ\bar{Y})]- \bar{Y})$$
# 

# ### 2. Grafique el equilibrio simultáneo en el modelo DA-OA.

# In[2]:


#1--------------------------
    # Demanda Agregada
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#2--------------------------
    # Oferta Agregada
    
# Parámetros

Y_size = 100

Pe = 100 
θ = 3
_Y = 20   

Y = np.arange(Y_size)


# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)


# In[3]:


# líneas punteadas autómaticas

# definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

    # coordenadas de las curvas (x,y)
A = [P_AD[0], Y[0]] # DA, coordenada inicio
B = [P_AD[-1], Y[-1]] # DA, coordenada fin

C = [P_AS[0], Y[0]] # L_45, coordenada inicio
D = [P_AS[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# In[4]:


# Gráfico del modelo DA-OA

# Dimensiones del gráfico
y_max = np.max(P)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar 
ax.plot(Y, P_AD, label = "DA", color = "C9") #DA
ax.plot(Y, P_AS, label = "OA", color = "C6") #OA

# Líneas punteadas
plt.axhline(y=intersec[0], xmin= 0, xmax= 0.5, linestyle = ":", color = "lightcoral")
plt.axvline(x=intersec[1],  ymin= 0, ymax= 0.49, linestyle = ":", color = "lightcoral")

# Texto agregado
plt.text(0, 200, '$P_0$', fontsize = 13, color = 'black')
plt.text(53, 25, '$Y_0$', fontsize = 13, color = 'black')
plt.text(50, 202, '$E_0$', fontsize = 13, color = 'black')


# Eliminar valores de ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Título, ejes y leyenda
ax.set(title="Equilibrio DA-OA", xlabel='Y', ylabel='P')
ax.legend()

plt.show()


# ### 3. Estática comparativa

# #### 3.1. Disminución del gasto fiscal ($ΔG_0 < 0$). Análisis intuitivo, matemático y gráfico. 

# Intuición:
# 
# - Modelo IS-LM: 
# 
# $$ G_0↓ → DA↓ → DA < Y → Y↓ $$
# 
# $$ Y↓ → M^d↓ → M^d < M^s → r↓$$
# 
# - Modelo DA-OA: 
# 
# $$ Y↓ → [P^e + θ(Y↓ - \bar{Y}])↓ → [P^e + θ(Y - \bar{Y})] > P → P↓$$

# Matemáticamente

# In[5]:


# nombrar variables como símbolos de IS
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos de LM 
k, j, Ms, P, Y = symbols('k j Ms P Y')

# nombrar variables como símbolos para curva de oferta agregada
Pe, _Y, Y, θ = symbols('Pe, _Y, Y, θ')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio en el modelo DA-OA
Y_eq = ( (1)/(θ + ( (j*beta_1+h*k)/h) ) )*( ( (h*Ms+j*beta_0)/h ) - Pe + θ*_Y )

# Precio de equilibrio en el modelo DA-OA 
P_eq = Pe + θ*(Y_eq - _Y)

# Tasa de interés de equilibrio en el modelo DA-OA
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms - P_eq)
#((h*Ms+j*beta_0)/h) - ((j*beta_1+h*r)/h)*((P-Pe-θ*_Y)/θ)


# In[6]:


# Efecto del cambio en el gasto autónomo sobre el producto en el modelo DA-OA
df_Y_eq_Pe = diff(Y_eq, Go)
print("El Diferencial del Producto con respecto al diferencial del precio esperado = ", df_Y_eq_Pe)


# 
# - Cambio del gasto autónomo sobre el producto $Y^e$ en el modelo DA-OA:
# 
# $$\frac{ΔY^e}{ΔG_0}= (+)$$
# 
# $$\frac{ΔY^e}{(-)}= (+)$$
# 
# $$ΔY^e= (-)$$

# In[7]:


# Efecto del cambio del gasto autónomo sobre la tasa de Interés en el modelo DA-OA
df_r_eq_Pe = diff(r_eq, Go)
print("El Diferencial del nivel de precios con respecto al diferencial del precio esperado = ", df_r_eq_Pe)


# 
# - Cambio del gasto autónomo sobre la tasa de interés $r^e$ en el modelo DA-OA:
# 
# $$\frac{Δr^e}{ΔG_0}= (+)$$
# 
# $$\frac{Δr^e}{(-)}= (+)$$
# 
# $$Δr^e= (-)$$

# In[8]:


# Efecto del cambio del gasto autónomo sobre el nivel de precios en el modelo DA-OA
df_P_eq_Pe = diff(P_eq, Go)
print("El Diferencial del nivel de precios con respecto al diferencial del precio esperado = ", df_P_eq_Pe)


# 
# - Cambio del gasto autónomo sobre el nivel de precios $P^e$ en el modelo DA-OA:
# 
# $$\frac{ΔP^e}{ΔG_0}= (+)$$
# 
# $$\frac{ΔP^e}{(-)}= (+)$$
# 
# $$ΔP^e= (-)$$

# Gráfico

# In[9]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)
#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 8           

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)
#--------------------------------------------------
    # NUEVA curva IS: reducción del gasto autónomo (Go)

# Definir SOLO el parámetro cambiado
Go = 41

# Generar la ecuación con el nuevo parámetro
def r_IS_G(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G= r_IS_G(b, m, t, Co, Io, Go, Xo, h, Y)
#---------------------------------------------------
# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "#400080") #IS
ax.plot(Y, i, label="LM", color = "C9")  #LM
ax.plot(Y, r_G, label="IS_G", color = "#9370db", linestyle ='dashed')  #IS_Go

# Título, ejes y leyenda
ax.set(title="IS-LM", xlabel='Y', ylabel='r')
ax.legend()



plt.show()


# In[10]:


#1--------------------------
    # Demanda Agregada ORGINAL
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)


# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#2--------------------------
  # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 70
θ = 3
_Y = 20  

Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)
#--------------------------------------------------
# NUEVA Demanda Agregada

# Definir SOLO el parámetro cambiado

Go = 32.5

# Generar la ecuación con el nuevo parámetro

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD_Go(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_Go = P_AD_Go(h, Ms, j, B0, B1, k, Y)
#------------------------------------
# Gráfico del modelo DA-OA
# Dimensiones del gráfico

y_max = np.max(P)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar 
ax.plot(Y, P_AD, label = "DA", color = "#006ad1") #DA
ax.plot(Y, P_AS, label = "OA", color = "C4") #OA
ax.plot(Y, P_Go, label = "DA_Go", color = "#00bfff", linestyle = 'dashed') #DA_Go

# Eliminar valores de ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Título, ejes y leyenda
ax.set(title="DA-OA", xlabel='Y', ylabel='P')
ax.legend()

plt.show()


# In[11]:


# líneas punteadas autómaticas
# definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

# coordenadas de las curvas (x,y)
A = [r_G[0], Y[0]] # DA, coordenada inicio
B = [r_G[-1], Y[-1]] # DA, coordenada fin

C = [i[0], Y[0]] # L_45, coordenada inicio
D = [i[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# In[12]:


# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: IS-LM
    
ax1.plot(Y, r, label = "IS", color = "#400080") #IS
ax1.plot(Y, i, label="LM", color = "C9")  #LM
ax1.plot(Y, r_G, label="IS_G", color = "#9370db", linestyle ='dashed')  #IS_Go

ax1.axvline(x=53.,  ymin= 0, ymax= 0.54, linestyle = ":", color = "grey") #azul punteada
ax1.axvline(x=56.9,  ymin= 0, ymax= 0.56, linestyle = ":", color = "grey") #anaranjada
ax1.axhline(y=81.9,  xmin= 0, xmax= 0.55, linestyle = ":", color = "grey") # azul
ax1.axhline(y=88.9,  xmin= 0, xmax= 0.57, linestyle = ":", color = "grey")

ax1.text(53, 25, '←', fontsize=15, color='blue')
ax1.text(76, 54, '←', fontsize=14, color='blue')
ax1.text(0, 84, '↓', fontsize=13, color='blue')
ax1.text(58, -25, '$Y_0$', fontsize=12, color='black')
ax1.text(45, -25, '$Y_1$', fontsize=12, color='#9370db')
ax1.text(0, 95, '$r_0$', fontsize=12, color='black')
ax1.text(0, 73, '$r_1$', fontsize=12, color='#9370db')
ax1.text(73, 60, '$ΔG_0$', fontsize=11, color='blue')

ax1.set(title="Efectos de una disminución del gasto fiscal (ΔG_0<0)", xlabel='Y', ylabel='r')
ax1.legend()
#---------------------------------
    # Gráfico 2:

ax2.plot(Y, P_AD, label = "DA", color = "#006ad1") #DA
ax2.plot(Y, P_AS, label = "OA", color = "C4") #OA
ax2.plot(Y, P_Go, label = "DA_Go", color = "#00bfff", linestyle = 'dashed') #DA_Go

ax2.axvline(x=52.5,  ymin= 0, ymax= 1, linestyle = ":", color = "grey") #verde
ax2.axvline(x=56.02,  ymin= 0, ymax= 1, linestyle = ":", color = "grey") #morada
ax2.axhline(y=168.3,  xmin= 0, xmax= 0.52, linestyle = ":", color = "grey") #verde
ax2.axhline(y=178.07,  xmin= 0, xmax= 0.56, linestyle = ":", color = "grey") #morada

ax2.text(53, 100, '←', fontsize=13, color='red')
ax2.text(73, 100, '←', fontsize=15, color='red')
ax2.text(0, 168, '↓', fontsize=12, color='red')

ax2.text(58, 0, '$Y_0$', fontsize=12, color='black')
ax2.text(45, 0, '$Y_1$', fontsize=12, color='#00bfff')
ax2.text(0, 180, '$P_0$', fontsize=12, color='black')
ax2.text(0, 155, '$P_1$', fontsize=12, color='#00bfff')
ax2.text(70, 110, '$ΔG_0$', fontsize=11, color='black')

ax2.set(xlabel='Y', ylabel='P')
ax2.legend()


plt.show


# #### 3.2. Disminución de la masa monetaria ($ΔM^s_0 < 0$). Análisis intuitivo, matemático y gráfico.

# Intuición:
# 
# - Modelo IS-LM: 
# 
# $$ M^s_0↓ → M^s↓ → M^s < M^d → r↑ $$
# 
# $$ r↑ → I↓ → DA↓ → DA < Y → Y↓$$
# 
# - Modelo DA-OA: 
# 
# $$ Y↓ → [P^e +θ(Y↓ - \bar{Y}]) → [P^e +θ(Y - \bar{Y}])↓ → [P^e + θ(Y - \bar{Y})] < P → P↓$$
# 
# Matemáticamente:

# In[13]:


# nombrar variables como símbolos de IS
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos de LM 
k, j, Ms, P, Y = symbols('k j Ms P Y')

# nombrar variables como símbolos para curva de oferta agregada
Pe, _Y, Y, θ = symbols('Pe, _Y, Y, θ')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio en el modelo DA-OA
Y_eq = ((1)/(θ + ( (j*beta_1+h*k)/h) ) )*( ( (h*Ms+j*beta_0)/h ) - Pe + θ*_Y )

# Precio de equilibrio en el modelo DA-OA 
P_eq = Pe + θ*(Y_eq - _Y)

# Tasa de interés de equilibrio en el modelo DA-OA
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms - P_eq)
#((h*Ms+j*beta_0)/h) - ((j*beta_1+h*r)/h)*((P-Pe-θ*_Y)/θ)


# In[14]:


# Efecto del cambio de masa monetaria esperado sobre el producto en el modelo DA-OA
df_Y_eq_Ms = diff(Y_eq, Ms)
print("El Diferencial del Producto con respecto al diferencial de la masa monetaria = ", df_Y_eq_Ms)
print("\n")


# - Cambios en el producto de equilibrio:
# 
# $$\frac{ΔY^e}{ΔM^S_0}= (+)$$
# 
# $$\frac{ΔY^e}{(-)}= (+)$$
# 
# $$ΔY^e= (-)$$

# In[15]:


# Efecto del cambio en la masa monetaria sobre la tasa de Interés en el modelo DA-OA
df_r_eq_Ms = diff(r_eq, Ms)
print("El Diferencial de la tasa de interés con respecto al diferencial de la masa monetaria = ", df_r_eq_Ms)
print("\n")


# - Cambios en la tasa de interés de equilibrio:
# 
# $$\frac{Δr^e}{ΔM^s_0}= (-)$$
# 
# $$\frac{Δr^e}{(-)}= (-)$$
# 
# $$Δr^e= (+)$$

# In[16]:


# Efecto del cambio de masa monetaria sobre el nivel de precios en el modelo DA-OA
df_P_eq_Ms = diff(P_eq, Ms)
print("El Diferencial del nivel de precios con respecto al diferencial de la masa monetaria = ", df_P_eq_Ms)


# - Cambios en el nivel de precios: 
# 
# $$\frac{ΔP^e}{ΔM^s_0}= (+)$$
# 
# $$\frac{ΔP^e}{(-)}= (+)$$
# 
# $$P^e= (-)$$

# Gráfico

# In[17]:


# IS-LM

#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 205             
P  = 8           

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM: incremento en la Masa Monetaria (Ms)

# Definir SOLO el parámetro cambiado
Ms = -105

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[18]:


#DA-OA

#1--------------------------
    # Demanda Agregada ORGINAL
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

k = 2
j = 1                
Ms = 205             
P  = 8  

Y = np.arange(Y_size)
# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#--------------------------------------------------
    # NUEVA Demanda Agregada

# Definir SOLO el parámetro cambiado

Ms = 130

# Generar la ecuación con el nuevo parámetro

def P_AD_Ms(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_Ms = P_AD_Ms(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 70
θ = 3
_Y = 20  

Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)


# In[19]:


# líneas punteadas autómaticas

    # definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

    # coordenadas de las curvas (x,y)
A = [P_AD[0], Y[0]] # DA, coordenada inicio
B = [P_AD[-1], Y[-1]] # DA, coordenada fin

C = [P_AS[0], Y[0]] # L_45, coordenada inicio
D = [P_AS[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# In[20]:


# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: IS-LM
    
ax1.plot(Y, r, label = "IS", color = "C3") #IS
ax1.plot(Y, i, label="LM", color = "C0")  #LM
ax1.plot(Y, i_Ms, label="LM_Ms", color = "C0", linestyle ='dashed')  #LM

ax1.axvline(x=45.15,  ymin= 0, ymax= 1, linestyle = ":", color = "lightcoral")
ax1.axvline(x=57,  ymin= 0, ymax= 1, linestyle = ":", color = "lightcoral")
ax1.axhline(y=88.9,  xmin= 0, xmax= 0.56, linestyle = ":", color = "lightcoral")
ax1.axhline(y=103.44,  xmin= 0, xmax= 0.46, linestyle = ":", color = "lightcoral")

ax1.text(75, 150, '∆$M_s$', fontsize=12, color='black')
ax1.text(76, 140, '←', fontsize=15, color='grey')
ax1.text(50, 50, '←', fontsize=15, color='grey')
ax1.text(0, 93, '↑', fontsize=15, color='grey')
ax1.text(60, 0, '$Y_0$', fontsize=12, color='black')
ax1.text(40, 0, '$Y_1$', fontsize=12, color='C0')
ax1.text(0, 75, '$r_0$', fontsize=12, color='black')
ax1.text(0, 110, '$r_1$', fontsize=12, color='C0')

ax1.set(title="Efectos de una reducción en la masa monetaria ($ΔM^s_0<0$)", xlabel='Y', ylabel='r')
ax1.legend()
#---------------------------------
    # Gráfico 2: DA-OA

ax2.plot(Y, P_AD, label = "DA", color = "C9") #DA
ax2.plot(Y, P_Ms, label = "DA_Ms", color = "C0", linestyle = 'dashed') #DA_Ms
ax2.plot(Y, P_AS, label = "OA", color = "C2") #OA

ax2.axvline(x=45,  ymin= 0, ymax= 1, linestyle = ":", color = "lightcoral")
ax2.axvline(x=56.9,  ymin= 0, ymax= 1, linestyle = ":", color = "lightcoral")
ax2.axhline(y=143.5,  xmin= 0, xmax= 0.45, linestyle = ":", color = "lightcoral")
ax2.axhline(y=179.5,  xmin= 0, xmax= 0.56, linestyle = ":", color = "lightcoral")

ax2.text(50, 30, '←', fontsize=15, color='grey')
ax2.text(36, 200, '←', fontsize=15, color='grey')
ax2.text(35, 212, '∆$M_s$', fontsize=12, color='black')
ax2.text(0, 155, '↓', fontsize=15, color='grey')

ax2.text(60, 0, '$Y_0$', fontsize=12, color='black')
ax2.text(40, 0, '$Y_1$', fontsize=12, color='C0')
ax2.text(0, 190, '$P_0$', fontsize=12, color='black')
ax2.text(0, 125, '$P_1$', fontsize=12, color='C0')

ax2.set(xlabel='Y', ylabel='P')
ax2.legend()

plt.show


# #### 3.3. Incremento de la tasa de impuestos $t>0$ .Análisis intuitivo, matemático y gráfico.

# Intuición:
# 
# - Modelo IS-LM: 
# 
# $$ t↑ → C↓ → DA↓ → DA < Y → Y↓$$
# 
# $$ Y↓ → M^d↓ → M^d < M^s → r↓ $$
# 
# - Modelo DA-OA: 
# 
# $$ Y↓ → [P^e +θ(Y↓ - \bar{Y})] → [P^e +θ(Y - \bar{Y})]↓ → [P^e + θ(Y - \bar{Y})] < P → P↓$$
# 
# Matemáticamente:

# In[21]:


# nombrar variables como símbolos de IS
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos de LM 
k, j, Ms, P, Y = symbols('k j Ms P Y')

# nombrar variables como símbolos para curva de oferta agregada
Pe, _Y, Y, θ = symbols('Pe, _Y, Y, θ')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio en el modelo DA-OA
Y_eq = ( (1)/(θ + ( (j*beta_1+h*k)/h) ) )*( ( (h*Ms+j*beta_0)/h ) - Pe + θ*_Y )

# Precio de equilibrio en el modelo DA-OA 
P_eq = Pe + θ*(Y_eq - _Y)

# Tasa de interés de equilibrio en el modelo DA-OA
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms - P_eq)
#((h*Ms+j*beta_0)/h) - ((j*beta_1+h*r)/h)*((P-Pe-θ*_Y)/θ)


# In[22]:


# Efecto del cambio de Precio esperado sobre Tasa de Interés en el modelo DA-OA
df_Y_eq_t = diff(Y_eq, t)
print("El Diferencial del Producto con respecto al diferencial de la masa monetaria = ", df_Y_eq_t)
print("\n")


# - Cambios en el producto:
# 
# $$\frac{ΔY^e}{Δt}= (-)$$
# 
# $$\frac{ΔY^e}{(+)}= (-)$$
# 
# $$Y^e= (-)$$

# In[23]:


# Efecto del cambio de tasa de impuestos sobre Tasa de Interés en el modelo DA-OA
df_r_eq_t = diff(r_eq, t)
print("El Diferencial de la tasa de interés con respecto al diferencial de la masa monetaria = ", df_r_eq_t)
print("\n")


# - Cambios en la tasa de interés de equilibrio:
# 
# $$\frac{Δr^e}{Δt}= (-)$$
# 
# $$\frac{Δr^e}{(+)}= (-)$$
# 
# $$r^e= (-)$$

# In[24]:


# Efecto del cambio de Precio esperado sobre Tasa de Interés en el modelo DA-OA
df_P_eq_t = diff(P_eq, t)
print("El Diferencial del nivel de precios con respecto al diferencial de la masa monetaria = ", df_P_eq_t)


# - Cambios en el nivel de precios:
# 
# $$\frac{ΔP^e}{Δt}= (-)$$
# 
# $$\frac{ΔP^e}{(+)}= (-)$$
# 
# $$P^e= (-)$$

# In[25]:


# IS-LM

#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.01

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)

#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros
Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 8           

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
# NUEVA curva IS: incremento en la tasa de impuestos (t)

# Definir SOLO el parámetro cambiado
t = 3.6

# Generar la ecuación con el nuevo parámetro
def r_IS_t(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t= r_IS_t(b, m, t, Co, Io, Go, Xo, h, Y)


# In[26]:


#DA-OA

#1--------------------------
    # Demanda Agregada ORIGINAL
    
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.01

k = 2
j = 1                
Ms = 200             
P  = 8  

Y = np.arange(Y_size)

# Ecuación

B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_AD = P_AD(h, Ms, j, B0, B1, k, Y)

#----------------------------------------
 # NUEVA Demanda Agregada

# Definir SOLO el parámetro cambiado

t = 6.8

# Generar la ecuación con el nuevo parámetro
B0 = Co + Io + Go + Xo
B1 = 1 - (b-m)*(1-t)

def P_AD_t(h, Ms, j, B0, B1, k, Y):
    P_AD = ((h*Ms + j*B0)/h) - (Y*(j*B1 + h*k)/h)
    return P_AD

P_t= P_AD_t(h, Ms, j, B0, B1, k, Y)


#2--------------------------
    # Oferta Agregada ORIGINAL
    
# Parámetros

Y_size = 100

Pe = 59
θ = 3
_Y = 20  

Y = np.arange(Y_size)

# Ecuación

def P_AS(Pe, _Y, Y, θ):
    P_AS = Pe + θ*(Y-_Y)
    return P_AS

P_AS = P_AS(Pe, _Y, Y, θ)


# In[27]:


# Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
# Gráfico 1: IS-LM
    
ax1.plot(Y, r, label = "IS", color = "#ff1493") #IS
ax1.plot(Y, i, label="LM", color = "#a284e0")  #LM
ax1.plot(Y, r_t, label="IS_t", color = "#ff7dbe", linestyle ='dashed')  #IS_t

ax1.axvline(x=58.78,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax1.axvline(x=51.40,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax1.axhline(y=92.60,  xmin= 0, xmax= 0.58, linestyle = ":", color = "grey")
ax1.axhline(y=77.80,  xmin= 0, xmax= 0.525, linestyle = ":", color = "grey")

ax1.text(76, 54, '∆$t$', fontsize=15, color='black')
ax1.text(76, 45, '←', fontsize=17, color='purple')
ax1.text(53, 15, '←', fontsize=15, color='purple')
ax1.text(0, 82, '↓', fontsize=15, color='purple')
ax1.text(60, 15, '$Y_0$', fontsize=13, color='black')
ax1.text(46, 15, '$Y_1$', fontsize=13, color='#ff7dbe')
ax1.text(0, 96, '$r_0$', fontsize=13, color='black')
ax1.text(0, 67, '$r_1$', fontsize=13, color='#ff7dbe')

ax1.set(title="Efectos de un aumento en la tasa de impuestos ($Δt>0$)", xlabel='Y', ylabel='r')
ax1.legend()
#---------------------------------
# Gráfico 2: DA-OA

ax2.plot(Y, P_AD, label = "DA", color = "#228b22") #DA
ax2.plot(Y, P_t, label = "DA_t", color = "#1ec71e", linestyle = 'dashed') #DA_t
ax2.plot(Y, P_AS, label = "OA", color = "#1e90ff") #OA

ax2.axvline(x=58.72,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axvline(x=51.57,  ymin= 0, ymax= 1, linestyle = ":", color = "grey")
ax2.axhline(y=175.68,  xmin= 0, xmax= 0.58, linestyle = ":", color = "grey")
ax2.axhline(y=153.74,  xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

#Flechas y diferenciales
ax2.text(53, 30, '←', fontsize=15, color='blue')
ax2.text(76.5, 63, '←', fontsize=17, color='blue')
ax2.text(0, 160, '↓', fontsize=15, color='blue')
ax2.text(76, 80, '∆$t$', fontsize=15, color='black')
ax2.text(60, 0, '$Y_0$', fontsize=13, color='black')
ax2.text(46, 0, '$Y_1$', fontsize=13, color='#1ec71e')
ax2.text(0, 187, '$P_0$', fontsize=13, color='black')
ax2.text(0, 135, '$P_1$', fontsize=13, color='#1ec71e')

ax2.set(xlabel='Y', ylabel='P')
ax2.legend()

plt.show


# In[28]:


# líneas punteadas autómaticas

    # definir la función line_intersection
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

    # coordenadas de las curvas (x,y)
A = [P_t[0], Y[0]] # DA, coordenada inicio
B = [P_t[-1], Y[-1]] # DA, coordenada fin

C = [P_AS[0], Y[0]] # L_45, coordenada inicio
D = [P_AS[-1], Y[-1]] # L_45, coordenada fin

    # creación de intersección

intersec = line_intersection((A, B), (C, D))
intersec # (y,x)


# ## Lectura

# El artículo Metas de Inflación en Perú: Las razones para el éxito escrito por Oscar Dancourt estudia la manera en que los principales instrumentos de la política monetaria peruana -la tasa de interés de referencia, el coeficiente de encaje legal en moneda nacional y extranjera y la intervención esterilizada en el mercado de divisas-fueron usados durante 2002-2013, e influenciaron la actividad económica y el nivel de precios por medio de 2 canales principales: el crédito y el tipo de cambio. De modo que, la pregunta de investigación del presente artículo es: ¿De qué manera fueron utilizados los principales instrumentos de política monetaria peruana y cuál fue su impacto en la actividad económica y el nivel de precios en el Perú durante el periodo 2002-2013? Ante esta pregunta, la investigación planteada establece que el BCRP tomó dos decisiones importantes en el periodo 2002-2013 con respecto a la política monetaria: la implementación de un sistema de metas de inflación con una tasa de interés de referencia como principal instrumento de política monetaria y la acumulación de suficientes reservas de divisas. Estas decisiones permitieron que se mantenga la estabilidad macroeconómica en contextos externos tanto favorables como desfavorables.
# 
# Sin embargo, es importante destacar que durante este periodo se contó mayormente con un contexto externo favorable a excepción de la crisis internacional en 2008-2009 y la crisis debido al ajuste de la política monetaria en 2013. Por tanto, para autenticar el importante rol de la política monetaria en la estabilidad macroeconómica del país durante dicho periodo, el autor la compara con la política monetaria para controlar la recesión de 1998-2000, pues en estos años la inflación no fue controlada de la misma manera que en las crisis de 2008-2009 y 2013.
# 
# La investigación tiene un enfoque mixto, pues además de presentar datos cuantitativos sobre la influencia de la política monetaria en la actividad económica en diversos contextos, también analiza el motivo por el cual el BCRP decidió aplicar distintas políticas monetarias con el pasar del tiempo, al comparar sus efectos en la economía. Así, a partir de este enfoque, se presentan ciertas ventajas y desventajas. En primer lugar, una ventaja del enfoque es que facilita la comprensión del objetivo principal del estudio, la cual es describir la manera en que se utilizaron los principales instrumentos de la política monetaria peruana en 2002-2013. Este objetivo se alcanza ya que, por medio de datos cuantitativos, gráficos estadísticos y explicaciones sencillas y detalladas, el lector comprende de forma más dinámica e integral los motivos detrás de las decisiones del BCRP sobre política monetaria y sus efectos en la actividad económica y el nivel de precios. En segundo lugar, otra ventaja del enfoque es que permite el uso de elementos teóricos (datos cualitativos) como es el caso del modelo de Bernanke – Blinder que explica de manera clara la influencia de la política monetaria en la economía peruana y al mismo tiempo reafirma las propuestas del autor, lo que brinda mayor validez a sus argumentos y a la investigación en general. En tercer lugar, una debilidad encontrada es que aún no es posible comprobar del todo la eficacia de los instrumentos de política monetaria para lograr y mantener la estabilidad macroeconómica en el Perú bajo un contexto externo desfavorable más prolongado, pues este en su mayoría fue favorable. Por tanto, esto se debe tomar en cuenta al poner en práctica los resultados del estudio.
# 
# A pesar de la debilidad presentada, no se puede negar la relevante contribución del artículo para el avance de la pregunta de investigación tanto a nivel teórico como práctico. Por un lado, la investigación contribuye teóricamente, pues brindan mayor conocimiento sobre la utilidad de una política monetaria menos restrictiva para lograr la estabilidad macroeconómica al resaltar el papel de sus principales instrumentos (la tasa de interés, la acumulación de reservas de divisas y la intervención esterilizada en el mercado de divisas) y al mostrar los efectos negativos de una política monetaria restrictiva en contextos externos desfavorables. Por otro lado, la investigación contribuye a nivel práctico, pues presenta recomendaciones para mantener la estabilidad macroeconómica y evitar posibles crisis económicas independientemente de si el contexto externo sea favorable o no. Algunas de ellas son la reducción de la tasa de interés de referencia y la venta esterilizada de dólares durante choques externos adversos.
# 
# Para finalizar, tal como menciona el autor algunos pasos próximos para avanzar en la pregunta serían analizar si el régimen de política monetaria planteado en dicho periodo funcionaría en un contexto externo más prolongado y verificar si este esquema puede operar sin la tasa de interés de referencia como principal instrumento del BCRP. Igualmente, ahora que ya se conoce los aspectos positivos de implementar un régimen de política monetaria con un esquema de metas de inflación y una tasa de referencia es conveniente analizar el aspecto negativo de este, pues tal como mencionan Schaefer y Velarde (2020), conducir este esquema de política monetaria si bien conlleva beneficios como la estabilización del nivel de precios y la reducción de la dolarización financiera, también significa ciertos riesgos e incertidumbre. Por último, el último paso que recomiendo sería comparar la política monetaria peruana con la de otros países emergentes que aplican políticas monetarias similares y presentan dificultades económicas inherentes para evaluar su eficacia.
# 
# - BIBLIOGRAFÍA:
# 
# Schaefer Luperdi, J. H., & Velarde García, C. I. (2020). Efectos del esquema de metas de inflación sobre el funcionamiento de la economía peruana.
# http://hdl.handle.net/11354/2651

# In[ ]:




