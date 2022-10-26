#!/usr/bin/env python
# coding: utf-8

# # REPORTE 4

# ## Lectura

# La investigación de Waldo Mendoza "Demanda y oferta agregada en presencia de políticas monetarias no convencionales" tiene como objetivo central modelar la política monetaria no convencional aplicada por el Banco Central de Estados Unidos, la Reserva Federal (Fed), para mitigar los efectos que dejó la crisis internacional del 2008 al 2009. Para esto, el autor parte del modelo convencional de oferta y demanda agregada de economía cerrada, el modelo IS-LM y lo complementa con los modelos con activos financieros de Tobin y Brainard (1963), Branson (1977) y Tobin (1981) para que así pueda añadir las innovaciones de la política monetaria americana en un lenguaje más sencillo y convincente. De modo que, la presente investigación busca responder la pregunta: ¿De qué manera se puede modelar la política monetaria no convencional aplicada por la Reserva Federal para mitigar los efectos de la crisis de 2008-2009 a partir de un modelo IS-LM combinado con modelos con activos financieros? Cómo contribución a la pregunta, en este artículo se mostrará por un lado que los antiguos modelos y métodos tradicionales macroeconómicos aún son útiles para analizar fenómenos macroeconómicos actuales y por otro lado, que las funciones de la Reserva Federal trascendieron producto de la política monetaria no convencional que aplicó frente a la crisis internacional de 2008-2009, pues “pasó de ser sólo un prestamista en cuestiones de última instancia a ser un creador de mercados de última instancia” (IMF, 2013).
# 
# Este estudio presenta un enfoque cuantitativo que parte de la formación de un modelo macroeconómico en base a la combinación de modelos que siguen la tradición del viejo keynesianismo: modelo IS-LM y los modelos con activos de Tobin y Brainard (1963), Branson (1977) y Tobin (1981). Así, a partir de este enfoque podemos distinguir algunas fortalezas y debilidades. En primer lugar, una fortaleza encontrada es que la incorporación de estos modelos con activos financieros complementando al tradicional modelo IS-LM hace que las innovaciones en la política monetaria americana sean presentadas de manera más sencilla y convincente para los lectores a comparación de los modelos y métodos macroeconómicos contemporáneos. En segundo lugar, debido a la combinación de los modelos macroeconómicos antes mencionados se crea un modelo IS-LM diferente y más completo para analizar dicho fenómeno macroeconómico, pues muestra a la Fed como administradora de la tasa de interés a corto plazo y presenta la adición de un mercado de bonos de largo plazo al modelo IS-LM. En conjunto, esto representa otra fortaleza, ya que se puede analizar los efectos de la política monetaria no convencional americana en distintos plazos (en el corto plazo, el tránsito hacia el equilibrio estacionario y el equilibrio estacionario) a partir de los valores de equilibrio de la producción, la tasa de interés de largo plazo y el nivel de precios. En tercer lugar, una debilidad que se debe considerar es que todo modelo tiene un margen de error, pues son simulaciones simplificadas de la realidad, mas no fieles reflejos de esta, de modo que no abarcan todas las variables contextuales que rodean al fenómeno macroeconómico, las cuales son igualmente importantes para explicar los efectos de la política monetaria no convencional americana.
# 
# Sin embargo, a pesar de la debilidad presentada, no se debe olvidar que la investigación contribuye de manera teórica y práctica con el avance de la pregunta. Por una parte, este artículo presenta un modelo construido a partir de viejos modelos tradicionales que es capaz de estudiar y explicar los efectos de las políticas macroeconómicas convencionales, no convencionales y los choques de oferta sobre la producción, el nivel de precios, la tasa de interés de largo plazo y la cantidad de dinero en términos sencillos, lo que conlleva un importante aporte teórico, ya que evidencia que los antiguos métodos económicos siguen siendo relevantes para estudiar fenómenos macroeconómicos actuales. Por otro lado, la investigación de Mendoza también contribuye a nivel práctico, pues el modelo puede servir como una guía para las autoridades de los países sobre los posibles efectos en su economía si deciden aplicar este tipo de políticas no convencionales, teniendo en cuenta que cada economía tiene sus particularidades.
# 
# Para finalizar, un paso próximo para avanzar la pregunta sería profundizar en las variables contextuales que dieron origen a las políticas monetarias no convencionales y si sería posible su funcionamiento en otras economías no solo en la estadounidense, pues en el contexto internacional se ha visto que muchas economías plantearon su uso e incluso ambiciosamente llegaron a aplicarlas incluso sin saber sus efectos. Tal como menciona Christine Lagarde (2013) se debe colaborar rápidamente para conocer más a fondo el impacto tanto local como internacional de las políticas no convencionales, para que no se desperdicie el tiempo en un círculo vicioso de ensayo y error. Adicionalmente, ahora que se sabe que los métodos económicos considerados obsoletos son útiles para resolver problemas macroeconómicos actuales, otro paso a seguir es comparar el nivel de efectividad de este modelo con el de modelos macroeconómicos contemporáneos, para así identificar cuál es más eficaz en cuanto a explicar las políticas no convencionales y sus efectos en la economía.
# 
# REFERENCIAS:
# Lagarde. A.(2013, 23 de agosto).El cálculo mundial de las políticas monetarias no convencionales. Fondo Monetario Internacional. https://www.imf.org/es/News/Articles/2015/09/28/04/53/sp082313
# 
# International Monetary Fund (IMF).(2013). Unconventional Monetary Policies - Recent Experiences and Prospects, Policy Papers.

# ## Código
# Roxana Rodriguez Pilco (20200373) y Greysi Arrelucea (20200279)

# In[1]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
#from causalgraphicalmodels import CausalGraphicalModel


# ### 1. Modelo IS-LM

# #### 1.1. Encuentre las ecuaciones de Ingreso ($Y^e$) y tasa de interes ($r^e$) de equilibrio (Escriba paso a paso la derivacion de estas ecuaciones).

# Para encontrar las ecuaciones de Ingreso ($Y^e$)  y tasa de interés ($r^e$) de equilibrio del modelo IS-LM se necesitan las ecuaciones de las curvas IS y LM:
# 
# Por un lado, la curva IS se deriva de la igualdad entre el ingreso ($Y$) y la demanda agregada ($DA$):
# 
# $$Y=C+I+G+X-M$$
# 
# Entonces, considerando las siguientes ecuaciones:
# 
# $$C= C_0 + bY^d$$
# 
# $$I= I_0 + hr$$
# 
# $$G= G_0$$
# 
# $$T= tY$$
# 
# $$X= X_0$$
# 
# $$M= mY^d$$
# 
# Para llegar al equilibrio Ahorro-Inversión, se debe restar la tributación ($T$) de ambos miembros de la igualdad:
# 
# $$Y-T = C+I+G+X-M-T$$ 
# 
# $$Y^d=C+I+G+X-M-T$$
# 
# $$Y^d-C-G-X+M+T = I$$
# 
# Esta igualdad se puede reescribir de la siguiente forma:
# 
# $$(Y^d-C)+(T-G)+(M-X)=I$$
# 
# Las tres partes de la derecha constituyen los tres componentes del ahorro total($S$): ahorro privado ($Sp=Y^d-C$) , ahorro del gobierno ($Sg=T-G$) y ahorro externo ($Se=M-X$):
# 
# $$S=Sp+Sg+Se$$
# 
# De modo que, el ahorro total ($S$) es igual a la inversión($I$):
# 
# $$Sp+Sg+Se=I$$
# 
# $$S(Y)=I(r)$$
# 
# Haciendo reemplazos se obtiene que:
# 
# $$Sp+Sg+Se=I_0-hr$$
# 
# $$(Y^d-C_0-bY^d)+(T-G_0)+(mY^d-X_0)=I_0-hr$$
# 
# Considerando las observaciones anteriores sobre los componentes de la condición de equilibrio ($Y$):
# 
# $$ [1 - (b - m)(1 - t)]Y - (C_0 + G_0 + X_0) = I_0 - hr $$
# 
# Entonces, la curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso:
# 
# $$ hr = (C_0 + G_0 + I_0 + X_0) - [1 - (b - m)(1 - t)]Y $$
# 
# luego, la sensibilidad de inversión ante la tasa de interés $h$ pasa al otro lado de la ecuación dividiendo todos los componentes y resulta:
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# Y puede simplificarse en:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Siendo $ B_0 = C_0 + G_0 + I_0 + X_0  $ el intercepto y la pendiente $  B_1 = 1 - (b - m)(1 - t) $
# 
# - Por otro lado, la ecuación de la curva LM:
# 
# El equilibrio en el Mercado de Dinero que da origen a la curva LM se genera del equilibrio entre la oferta de dinero $M^s$ y la demanda de dinero $M^d$, resultando:
# 
# $$M^s=M^d$$
# 
# Donde:
# 
# La oferta de dinero ($M^s$) → $ \frac{M^s}{P}=\frac{M^s_0}{P_0}$ 
# 
# La demanda de dinero ($M^d$) → $M_d=kY-jr$, pues se asume que la tasa de interés nominal es igual a la real $i=r$
# 
# Entonces, la igualdad de la demanda y la oferta de dinero resulta:
# 
# $$ \frac{M^s_0}{P_0}=kY-jr$$
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
# 
# Para encontrar el nivel de Ingresos de equilibrio ($Y^e$)  y la tasa de interés de equilibrio ($r^e$) se puede hacer por cualquiera de los tres métodos: reducción, igualación o sustitución.
# 
# Se igualan las curvas IS y LM:
# 
# $$-\frac{1}{j}\frac{M^s_0}{P_0}+\frac{k}{j}Y=\frac{B_0}{h} - \frac{B_1}{h}Y$$
# 
# Resultando:
# 
# - Ingreso de equilibrio ($Y^e$)
# 
# 
# $$Y^e=\frac{jB_0}{kh+jB_1}+(\frac{h}{kh+jB_1})\frac{M^s_0}{P_0}$$
# 
# 
# - Tasa de interés de equilibrio ($r^e$)
# 
# 
# $$r^e=\frac{kB_0}{kh+jB_1}+(\frac{B_1}{kh+jB_1})\frac{M^s_0}{P_0}$$
# 
# 
# Estas 2 ecuaciones representan el modelo IS-LM

# #### 1.2. Grafique el equilibrio simultáneo en los mercados de bienes y de dinero.

# In[2]:


# Curva IS

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
    r_IS = (Co + Io + Go + Xo)/h - ( ( 1-(b-m)*(1-t) ) / h)*Y  
    return r_IS

r_is = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación
def r_LM(k, j, Ms, P, Y):
    r_LM = - (1/j)*(Ms/P) + (k/j)*Y
    return r_LM

r_lm = r_LM( k, j, Ms, P, Y)


# In[3]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(r_lm)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
# Curva IS
ax.plot(Y, r_is, label = "IS", color = "#ffb3ff") #IS
# Curva LM
ax.plot(Y, r_lm, label="LM", color = "#ccccff")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
# Graficar la linea horizontal - r
plt.axvline(x=52.3,  ymin= 0, ymax= 0.522, linestyle = ":", color = "purple")
# Graficar la linea vertical - Y
plt.axhline(y=94, xmin= 0, xmax= 0.522, linestyle = ":", color = "purple")

# Plotear los textos 
plt.text(51,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(53,-10, '$Y_0$', fontsize = 12, color = 'black')
plt.text(49,130, '$I$', fontsize = 14, color = 'blue')
plt.text(75,82, '$II$', fontsize = 14, color = 'blue')
plt.text(48.5,40, '$III$', fontsize = 14, color = 'blue')
plt.text(25,82, '$IV$', fontsize = 14, color = 'blue')

# Título, ejes y leyenda
ax.set(title="Equilibrio simultáneo en los mercados de bienes y dinero", xlabel='Y', ylabel='r')
ax.legend()

plt.show()


# ### 2. Estática comparativa.

# #### 2.1. Analice los efectos sobre las variables endógenas Y, r de una disminución del gasto fiscal $(ΔG_0<0)$ . El análisis debe ser intuitivo, matemático y gráfico.

# ###### Análisis intuitivo
# 
# - Mercado de Bienes
# 
# $$ G_0↓ → DA↓ → DA<Y → Y↓$$
# 
# - Mercado de dinero
# 
# $$ Y↓ → Md↓ → Md<Ms → r↓$$

# ###### Análisis matemático

# In[4]:


# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[5]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[6]:


df_Y_eq_Go = diff(Y_eq, Go)
print("El Diferencial del Producto con respecto al diferencial del gasto autónomo = ", df_Y_eq_Go)  # este diferencial es positivo


# In[7]:


df_r_eq_Go = diff(r_eq, Go)
print("El Diferencial de la tasa de interés con respecto al diferencial del gasto autónomo = ", df_r_eq_Go)  # este diferencial es positivo


# - Cambios en el producto de equilibrio:
# 
# $$ΔY_e= \frac{j}{kh+jB_1}ΔG_0<0 → Y (-)$$
# 
# $$ΔY_e= (+)(-) < 0 →(-)$$
# 
# - Cambios en la tasa de interés de equilibrio:
# 
# $$Δr_e= \frac{k}{kh+jB_1}ΔG_0 < 0$$
# 
# $$Δr_e= (+)(-) < 0 →(-)$$

# ###### Gráfico

# In[8]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 79
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
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[9]:


#--------------------------------------------------
    # NUEVA curva IS: reducción Gasto de Gobienro (Go)
    
# Definir SOLO el parámetro cambiado
Go = 50

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[10]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "#00008b") #IS_orginal
ax.plot(Y, r_G, label = "IS_(G_1)", color = "#33ccff", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "#9370db")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=52.3,  ymin= 0, ymax= 0.505, linestyle = ":", color = "#ffb6c1")
plt.axhline(y=94.6, xmin= 0, xmax= 0.526, linestyle = ":", color = "#ffb6c1")
plt.axvline(x=63.5,  ymin= 0, ymax= 0.605, linestyle = ":", color = "#ffb6c1")
plt.axhline(y=117, xmin= 0, xmax= 0.625, linestyle = ":", color = "#ffb6c1")

#Títulos de las variables
plt.text(49,100, '$E_1$', fontsize = 14, color = 'black')
plt.text(61,122, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,120, '$r_0$', fontsize = 13, color = 'black')
plt.text(-1,98, '$r_1$', fontsize = 13, color = 'black')
plt.text(65,12, '$Y_0$', fontsize = 13, color = 'black')
plt.text(54,12, '$Y_1$', fontsize = 13, color = 'black')
plt.text(69, 84, '$ΔG_0$', fontsize=15, color='blue')
#Flechas
plt.text(70, 78, '←', fontsize=15, color='blue')
plt.text(56, 35, '←', fontsize=15, color='blue')
plt.text(4, 104, '↓', fontsize=15, color='blue')

# Título, ejes y leyenda
ax.set(title="Efectos de la disminución del Gasto Autónomo($G_0<0$)", xlabel='Y', ylabel='r')
ax.legend()

plt.show()


# #### 2.2. Analice los efectos sobre las variables endógenas Y, r de una disminución de la masa monetaria $(ΔM^s<0)$ . El análisis debe ser intuitivo, matemático y gráfico.

# Análisis intuitivo
# 
# - Mercado de dinero
# 
# $$ M^s_0↓ → M^s<M^d → r↑ $$
# 
# - Mercado de bienes
# 
# $$ r↑ → I↓ → DA↓ → DA<Y → Y↓ $$

# ###### Análisis matemático

# In[11]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[12]:


df_r_eq_Ms = diff(r_eq, Ms)
print("El Diferencial de la tasa de interés con respecto al diferencial de la masa monetaria = ", df_r_eq_Ms)  # este diferencial es positivo


# In[13]:


df_Y_eq_Ms = diff(Y_eq, Ms)
print("El Diferencial del producto con respecto al diferencial de la masa monetaria = ", df_Y_eq_Ms)  # este diferencial es positivo


# 
# - Cambios en la tasa de interés de equilibrio:
# 
# $$Δr_e= -\frac{B_1}{kh+jB_1}\frac{1}{P}ΔM^s_0 >0$$
# 
# $$Δr_e= (-)(+)(+)(-) > 0 →(+)$$
# 
# - Cambios en el ingreso de equilibrio:
# 
# $$\fracΔY_e= \frac{h}{kh+jB_1}\frac{1}{P}ΔM^s_0 < 0$$
# 
# $$\frac{ΔY_e}{ΔM^s_0}= (+)(+)(-) < 0 →(-)$$
# 

# ###### Gráfico

# In[14]:


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
Ms = 700             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[15]:


# Definir SOLO el parámetro cambiado, en este caso, disminuye la masa monetaria
Ms = 200

# Generar nueva curva LM con la variacion del Ms
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[16]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "#990099") #IS_orginal
ax.plot(Y, i, label="LM_(MS_0)", color = "#00cc66")  #LM_original

ax.plot(Y, i_Ms, label="LM_(MS_1)", color = "#80b8f0", linestyle = 'dashed')  #LM_modificada

# Lineas de equilibrio_1 
plt.axvline(x=52.3,  ymin= 0, ymax= 0.57, linestyle = ":", color = "grey")
plt.axhline(y=94.3, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

# Lineas de equilibrio_0 
plt.axvline(x=60,  ymin= 0, ymax= 0.53, linestyle = ":", color = "grey")
plt.axhline(y=84.7, xmin= 0, xmax= 0.59, linestyle = ":", color = "grey")

# Textos ploteados
plt.text(64,83, '$E_0$', fontsize = 14, color = 'black')
plt.text(-1,72, '$r_0$', fontsize = 12, color = 'black')
plt.text(61,-40, '$Y_0$', fontsize = 12, color = 'black')
plt.text(49,100, '$E_1$', fontsize = 14, color = 'black')
plt.text(-1,100, '$r_1$', fontsize = 12, color = 'black')
plt.text(48,-40, '$Y_1$', fontsize = 12, color = 'black')
plt.text(69, 120, '$ΔM^s_0$', fontsize=13, color='blue')
#Flechas
plt.text(70, 115, '←', fontsize=13, color='blue')
plt.text(55, 35, '←', fontsize=15, color='blue')
plt.text(4, 86, '↓', fontsize=15, color='blue')

# Título, ejes y leyenda
ax.set(title="Efectos de una Reducción de la Masa Monetaria ($M^s_0<0$)", xlabel='Y', ylabel='r')
ax.legend()

plt.show()


# #### 2.3. Analice los efectos sobre las variables endógenas Y, r de un incremento de la tasa de impuestos $(Δt>0)$ . El análisis debe ser intuitivo, matemático y gráfico.

# ###### Análisis intuitivo
# 
# - Mercado de bienes
# 
# $$ t↑ → DA↓ → DA<Y → Y↓ $$
# 
# - Mercado de dinero
# 
# $$ Y↓ → M^d<M^s → r↓ $$

# ###### Análisis matemático

# In[17]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
Y_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[18]:


df_r_eq_t = diff(r_eq, t)
print("El Diferencial de la tasa de interés con respecto al diferencial de la tasa impositiva = ", df_r_eq_t)  # este diferencial es negativo


# In[19]:


df_Y_eq_t = diff(Y_eq, t)
print("El Diferencial del producto con respecto al diferencial de la tasa impositiva = ", df_Y_eq_t)  # este diferencial es negativo


# 
# Tanto la tasa de interés como el producto se reducen, por ende su diferencial sería negativo y el diferencial de la tasa impositiva sería positivo ya que aumentó. Esto me dice que los diferenciales que aparecen en el código serían negativos.
# 
# - Cambios en el la tasa de interés de equilibrio:
# 
# $$Δr^e/Δt = (-)$$
# 
# $$Δr^e/(+) = (-)$$
# 
# $$Δr^e = (-)$$
# 
# - Cambios en el ingreso de equilibrio:
# 
# $$ΔY^e/Δt = (-)$$
# 
# $$ΔY^e/(+) = (-)$$
# 
# $$ΔY^e = (-)$$

# ###### Gráfico

# In[20]:


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
t = 0.03

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS= (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[21]:


#--------------------------------------------------
    # NUEVA curva IS: aumento de tasa impositiva (t)
    
# Definir SOLO el parámetro cambiado
t = 0.99

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS= (Co + Io + Go + Xo -Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[22]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(t_0)", color = "#ff5079") #IS_orginal
ax.plot(Y, r_t, label = "IS_(t_1)", color = "#20b2aa", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "#3399ff")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=52,  ymin= 0, ymax= 0.53, linestyle = ":", color = "#808080")
plt.axhline(y=93.5, xmin= 0, xmax= 0.53, linestyle = ":", color = "#808080")
plt.axvline(x=54,  ymin= 0, ymax= 0.54, linestyle = ":", color = "#808080")
plt.axhline(y=97.5, xmin= 0, xmax= 0.54, linestyle = ":", color = "#808080")

#Títulos de las variables
plt.text(49,82, '$E_1$', fontsize = 12, color = 'black')
plt.text(52,102, '$E_0$', fontsize = 12, color = 'black')
plt.text(-1,99, '$r_0$', fontsize = 12, color = 'black')
plt.text(-1,88, '$r_1$', fontsize = 12, color = 'black')
plt.text(54,10, '$Y_0$', fontsize = 12, color = 'black')
plt.text(49,10, '$Y_1$', fontsize = 12, color = 'black')
plt.text(83, 58.5, '$Δt$', fontsize=11, color='black')
#Flechas
plt.text(83, 55, '←', fontsize=12, color='black')
plt.text(52, 35, '←', fontsize=12, color='blue')
plt.text(4, 94, '↓', fontsize=12, color='blue')

# Título, ejes y leyenda
ax.set(title="Efectos del aumento de la Tasa Impositiva($t>0$)", xlabel='Y', ylabel='r')
ax.legend()

plt.show()


# ## Código (Tres puntos Extra!)

# ### 1. Modelo IS-LM

# #### 1.1. Encuentre las ecuaciones de Ingreso $(Y^e)$ y tasa de interes $(r^e)$ de equilibrio(Escriba paso a paso la derivacion de estas ecuaciones).

# 
# Para obtener la curva IS, se parte de la identidad Ingreso-Costo y se busca llegar al equilibrio ahorro-inversión
# 
# $$Y=C+I+G$$
# 
# Para llegar al equilibrio Ahorro-Inversión, se debe restar la tributación ($T$) de ambos miembros de la identidad
# 
# $$Y=C_0+bY^d+I_0-hr+G_0$$
# 
# $$Y=C_0+bY^d+I_0-hr+G_0$$
# 
# $$hr=C_0+I_0+G_0+b(1-t)Y-Y$$
# 
# $$ hr = (C_0 + G_0 + I_0) - [1 - b(1 - t)]Y $$
# 
# Entonces, la ecuación de la curva IS es:
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0) - \frac{1 - b(1 - t)}{h}Y $$
# 
# De manera reducida:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $B_0=C_0 + G_0 + I_0 y B_1=1 - b(1 - t)$
# 
# La ecuación de la curva LM surge de la igualdad entre la oferta de dinero y la demanda de dinero
# 
# $$ \frac{M^s_0}{P_0} = kY - j(r + π^e) $$
# 
# $$ j(r + π^e) = kY - \frac{M^s_0}{P_0} $$
# 
# $$ r + π^e = - \frac{M^s_0}{jP_0} + \frac{kY}{j} $$
# 
# Entonces, la ecuación de la curva LM es:
# 
# $$ r = - \frac{M^s_0}{jP_0} + \frac{k}{j}Y - π^e $$
# 
# Por tanto, para hallar las ecuaciones de ingreso ($Y^e$) y tasa de interés de equilibrio ($r^e$), se igualan las ecuaciones de las curvas IS y LM
# 
# - Hallar $Y^e$
# 
# $$ \frac{B_0}{h} - \frac{B_1}{h}Y = - \frac{M^s_0}{jP_0} + \frac{k}{j}Y - π^e $$
# 
# $$ \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e = \frac{k}{j}Y + \frac{B_1}{h}Y $$
# 
# $$ Y(\frac{k}{j} + \frac{B_1}{h}) = \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e $$
# 
# $$ Y(\frac{hk + jB_1}{jh}) = \frac{B_0}{h} + \frac{M^s_0}{jP_0} + π^e $$
# 
# Resultando:
# 
# $$ Y^e = \frac{jB_0}{kh + jB_1} + \frac{M_0^s}{P_0} \frac{h}{kh + jB_1} + \frac{jh}{kh + jB_1} π^e $$
# 
# - Hallar $r^e$
# 
# $$ r^e = - \frac{Ms_o}{P_o} (\frac{B_1}{kh + jB_1}) + \frac{kB_o}{kh + jB_1} - \frac{B_1}{kh + jB_1} π^e $$

# #### 1.2. Grafique el equilibrio simultáneo en los mercados de bienes y de dinero.

# In[23]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2

j = 1                
Ms = 200             
P  = 20
π = 4

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


# In[24]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "C1") #IS
ax.plot(Y, i_2, label="LM", color = "C0")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=55,  ymin= 0, ymax= 0.54, linestyle = ":", color = "black")
plt.axhline(y=94, xmin= 0, xmax= 0.55, linestyle = ":", color = "black")
plt.text(53,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-15, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Equilibrio modelo IS-LM", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# In[25]:


# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# ### 2. Estática comparativa.
# #### 2.1. Analice los efectos sobre las variables endógenas Y, r de una disminución de los Precios $(ΔP_0<0)$. El análisis debe ser intuitivo, matemático y gráfico.

# Análisis intuitivo
# 
# - Mercado de dinero
# 
# $$ P_0↓ →  P↓ → M^s>M^d → r↓ $$
# 
# - Mercado de bienes
# 
# $$ r↓ → I↑ → DA↑ → DA>Y → Y↑ $$

# Análisis matemático

# In[26]:


# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# In[27]:


df_Y_eq_P = diff(Y_eq, P)
print("El Diferencial del Producto con respecto al diferencial del nivel de precios = ", df_Y_eq_P)


# 
# - Cambios en $Y^e$:
# 
# ¿$∆Y$ sabiendo que $∆P < 0$?
# 
# $$ \frac{∆Y}{∆P} = (-) $$
# 
# $$ \frac{∆Y}{(-)} = (-) $$
# 
# $$ ∆Y > 0 $$

# In[28]:


df_r_eq_P = diff(r_eq, P)
print("El Diferencial de la tasa de interés con respecto al diferencial del nivel de precios = ", df_r_eq_P)


# 
# - Cambios en $r^e$:
# 
# ¿$∆r$ sabiendo que $∆P < 0$?
# 
# $$ \frac{∆r}{∆P} = (+) $$
# 
# $$ \frac{∆r}{(-)} = (+) $$
# 
# $$ ∆r < 0 $$
# 
# - Gráfico

# In[29]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20
π = 4

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


#--------------------------------------------------
    # Nueva curva LM 
    
P = 5

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2_P = i_LM_2( k, j, Ms, P, Y, π)


# In[30]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "C3") #IS
ax.plot(Y, i_2, label="LM", color = "C0")  #LM
ax.plot(Y, i_2_P, label="LM", color = "C9", linestyle ='dashed')  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=55,  ymin= 0, ymax= 0.6, linestyle = ":", color = "black")
plt.axhline(y=94, xmin= 0, xmax= 0.55, linestyle = ":", color = "black")
plt.text(53,102, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-45, '$Y_0$', fontsize = 12, color = 'black')

plt.axvline(x=64.5,  ymin= 0, ymax= 0.56, linestyle = ":", color = "black")
plt.axhline(y=85, xmin= 0, xmax= 0.64, linestyle = ":", color = "black")
plt.text(62,90, '$E_1$', fontsize = 14, color = 'C0')
plt.text(0,75, '$r_1$', fontsize = 12, color = 'C0')
plt.text(66,-45, '$Y_1$', fontsize = 12, color = 'C0')

# Título, ejes y leyenda
ax.set(title="Disminución del Precio", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# #### 2.2. Analice los efectos sobre las variables endógenas Y, r de una disminución de la inflación esperada $(∆π < 0)$. El análisis debe ser intuitivo, matemático y gráfico.

# 
# - Análisis intuititvo
# 
# $$ π↓ → r↑ $$
# 
# $$ r↑ → I↓ → DA↓ → DA < Y → Y↓ $$

# In[31]:



# nombrar variables como símbolos
Co, Io, Go, h, r, b, t, beta_0, beta_1  = symbols('Co, Io, Go, h, r, b, t, beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y, π = symbols('k j Ms P Y π')

# Beta_0 y beta_1
beta_0 = (Co + Io + Go)
beta_1 = (1 - b*(1-t))

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
r_eq = -(Ms/P)*(beta_1/(k*h+j*beta_1)) + ((k*beta_0)/k*h+j*beta_1) - ((beta_1*π)/k*h+j*beta_1)
Y_eq = ((j*beta_0)/(k*h+j*beta_1)) + (Ms/P)*(h/(k*h+j*beta_1)) + (j*h*π/(k*h+j*beta_1))


# In[32]:


df_Y_eq_π = diff(Y_eq, π)
print("El Diferencial del Producto con respecto al diferencial del nivel de inflación = ", df_Y_eq_π)


# Análisis matemático:
# 
# - Cambios en $Y^e$
# 
# ¿$∆Y$ sabiendo que $∆π < 0$?
# 
# $$ \frac{∆Y}{∆π} = (+) $$
# 
# $$ \frac{∆Y}{(-)} = (+) $$
# 
# $$ ∆Y < 0 $$

# In[33]:


df_r_eq_π = diff(r_eq, π)
print("El Diferencial de la tasa de interés con respecto al diferencial del nivel de inflación = ", df_r_eq_π)


# 
# - Cambios en $r_eq$
# 
# ¿$∆r$ sabiendo que $∆π < 0$?
# 
# $$ \frac{∆r}{∆π} = (-) $$
# 
# $$ \frac{∆r}{(-)} = (-) $$
# 
# $$ ∆r > 0 $$
# 
# - Gráfico

# In[34]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
h = 0.8
b = 0.5
t = 0.8

Y = np.arange(Y_size)

# Ecuación 
def r_IS_2(b, t, Co, Io, Go, h, Y):
    r_IS_2 = (Co + Io + Go - Y * (1-b*(1-t)))/h
    return r_IS_2

r_2 = r_IS_2(b, t, Co, Io, Go, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20
π = 20

Y = np.arange(Y_size)

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2 = i_LM_2( k, j, Ms, P, Y, π)


#--------------------------------------------------
    # Nueva curva LM 
    
π = 2

# Ecuación

def i_LM_2(k, j, Ms, P, Y, π):
    i_LM_2 = (-Ms/P)/j + k/j*Y - π
    return i_LM_2

i_2_π = i_LM_2( k, j, Ms, P, Y, π)


# In[35]:


# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r_2, label = "IS", color = "C5") #IS
ax.plot(Y, i_2, label="LM", color = "C0")  #LM
ax.plot(Y, i_2_π, label="LM", color = "C9", linestyle ='dashed')  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
plt.axvline(x=54,  ymin= 0, ymax= 0.57, linestyle = ":", color = "black")
plt.axhline(y=95, xmin= 0, xmax= 0.54, linestyle = ":", color = "black")
plt.text(52,103, '$E_1$', fontsize = 14, color = 'C0')
plt.text(0,100, '$r_1$', fontsize = 12, color = 'C0')
plt.text(50,-35, '$Y_1$', fontsize = 12, color = 'C0')

plt.axvline(x=60,  ymin= 0, ymax= 0.55, linestyle = ":", color = "black")
plt.axhline(y=89, xmin= 0, xmax= 0.6, linestyle = ":", color = "black")
plt.text(58,95, '$E_0$', fontsize = 14, color = 'black')
plt.text(0,80, '$r_0$', fontsize = 12, color = 'black')
plt.text(56,-35, '$Y_0$', fontsize = 12, color = 'black')

# Título, ejes y leyenda
ax.set(title="Disminución de la inflación esperada", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# In[ ]:




