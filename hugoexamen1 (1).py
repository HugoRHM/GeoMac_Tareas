#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Función a ser evaluada
y = lambda x: 1 - np.exp(1)**x + (np.exp(1) - 1) * np.sin(np.pi*x / 2)

# Dominio de la función
xmin = 0
xmax = 4
x = np.linspace(xmin, xmax, 100)

A = 0  # Extremo izquierdo del intervalo
B = 1  # Extremo derecho del intervalo

# Gráfica de la función
plt.plot(x,y(x), 'r-', lw=2)

# Líneas verticales en los extremos del intervalo y línea en y = 0
ymin = np.min(y(x))
ymax = np.max(y(x))
plt.plot([A,A], [ymin,ymax], 'g--', lw=1)
plt.plot([B,B], [ymin,ymax], 'g--', lw=1)
plt.plot([xmin, xmax], [0,0], 'b-', lw=1)

plt.grid()
plt.show()


# In[60]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Función a ser evaluada
y = lambda x: (x - 1) * np.tan(x) + x * np.sin(np.pi * x)

# Dominio de la función
xmin = 0
xmax = 4
x = np.linspace(xmin, xmax, 100)

A = 1  # Extremo izquierdo del intervalo
B = 2  # Extremo derecho del intervalo

# Gráfica de la función
plt.plot(x,y(x), 'r-', lw=2)

# Líneas verticales en los extremos del intervalo y línea en y = 0
ymin = np.min(y(x))
ymax = np.max(y(x))
plt.plot([A,A], [ymin,ymax], 'g--', lw=1)
plt.plot([B,B], [ymin,ymax], 'g--', lw=1)
plt.plot([xmin, xmax], [0,0], 'b-', lw=1)

plt.grid()
plt.show()


# In[69]:


import numpy as np
import matplotlib.pyplot as plt
from math import log 
get_ipython().run_line_magic('matplotlib', 'inline')

# Función a ser evaluada
y = lambda x: x * np.sin(np.pi * x) - (x - 2) * np.log(x)

# Dominio de la función
xmin = 0
xmax = 4
x = np.linspace(xmin, xmax, 100)

A = 1  # Extremo izquierdo del intervalo
B = 2  # Extremo derecho del intervalo

# Gráfica de la función
plt.plot(x,y(x), 'r-', lw=2)

# Líneas verticales en los extremos del intervalo y línea en y = 0
ymin = np.min(y(x))
ymax = np.max(y(x))
plt.plot([A,A], [ymin,ymax], 'g--', lw=1)
plt.plot([B,B], [ymin,ymax], 'g--', lw=1)
plt.plot([xmin, xmax], [0,0], 'b-', lw=1)

plt.grid()
plt.show()


# In[85]:


import numpy as np
import matplotlib.pyplot as plt
from math import log 
get_ipython().run_line_magic('matplotlib', 'inline')

# Función a ser evaluada
y = lambda x: (x - 2) * np.sin(x) * np.log(x + 2)

# Dominio de la función
xmin = -2
xmax = 8
x = np.linspace(xmin, xmax, 100)

A = 0  # Extremo izquierdo del intervalo
B = 2  # Extremo derecho del intervalo

# Gráfica de la función
plt.plot(x,y(x), 'r-', lw=2)

# Líneas verticales en los extremos del intervalo y línea en y = 0
ymin = np.min(y(x))
ymax = np.max(y(x))
plt.plot([A,A], [ymin,ymax], 'g--', lw=1)
plt.plot([B,B], [ymin,ymax], 'g--', lw=1)
plt.plot([xmin, xmax], [0,0], 'b-', lw=1)

plt.grid()
plt.show()


# In[35]:


Ea= np.fabs(np.pi - 22/7)
print(Ea)

Er = Ea / np.fabs(np.pi)
print(Er)

print('Error absoluto = {: .20}'.format(Ea))
print('Error relativo = {: .20}'.format(Er))


# In[36]:


Ea= np.fabs(np.exp(10) - 22000)
print(Ea)

Er = Ea / np.fabs(np.exp(10))
print(Er)

print('Error absoluto = {: .20}'.format(Ea))
print('Error relativo = {: .20}'.format(Er))


# In[37]:


Ea= np.fabs(10**np.pi - 1400)
print(Ea)

Er = Ea / np.fabs(10**np.pi)
print(Er)

print('Error absoluto = {: .20}'.format(Ea))
print('Error relativo = {: .20}'.format(Er))


# In[58]:


from math import sqrt
Ea= np.fabs(np.math.factorial(9) - np.sqrt(18 * np.pi * (9 / np.exp(1))**9)
print(Ea)
            
Er = Ea / np.math.factorial(9)
print(Er)

print('Error absoluto = {: .20}'.format(Ea))
print('Error relativo = {: .20}'.format(Er))


# In[84]:


uno = np.fabs(np.math.factorial(9))
print(Ea)

dos = np.sqrt(18 * np.pi * (9 / np.exp(1))**9)
print(Er)

Ea = uno - dos 
print(Ea)

Er = Ea / np.fabs(np.math.factorial(9))

print('Error absoluto = {: .20}'.format(Ea))
print('Error relativo = {: .20}'.format(Er))


# In[81]:


#Funcion 1
f = lambda x : np.exp(x)

#Metodo a
#Taylor de  0 a 9

ma = lambda x : (((-1)*i * 5*i)/np.math.factorial(i))

#Sumando a
suma1=0
for i in range(0,9):
    
    ma = (((-1)*i * 5*i)/np.math.factorial(i))
    suma1= suma1 + ma
    #print(ma)
print('Aproximacion metodo a =', suma1)
print('')
#Metodo b

mb = lambda x : (1)/(5)**i / np.math.factorial(i)

#Sumando b
suma2=0
for i in range(0,9):
    
    mb = (1)/(5)**i * np.math.factorial(i)
    suma2 = suma2 + mb
    #print(mb)
print('Aproximacion metodo b =', suma2)


# In[113]:


#para aproximar a 0.0673
f = lambda x : np.exp(x)

#Metodo a
#i va de 0 hasta 9

ma = lambda x : (((-1)**i * 5**i)/np.math.factorial(i))

#Sumando a
suma1=0
for i in range(0,21):
    
	ma = (((-1)**i * 5**i)/np.math.factorial(i))
	suma1= suma1 + ma
	#print(ma)
print('Aproximacion metodo a')
print (suma1)
print('')
#Metodo b

mb = lambda x : (1)/(5)**i / np.math.factorial(i)

#Sumando b
suma2=0
for i in range(0,1):
    
	mb = ((1)/(((5)**i) / (np.math.factorial(i))))
	suma2 = suma2 + mb
	#print(mb)
print('Aproximacion metodo b')
print (suma2)

