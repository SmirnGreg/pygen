import math
import numpy as np
import matplotlib.pyplot as plt
import random


def mysin(x, p):
    return p[0] * np.sin(p[1] * x)


def residuals(datax, datay, params, model=mysin):
    """I work!"""
    return np.sum((datay - model(datax, params)) ** 2)


# Defining initial data

p0 = np.array([1.5, 1.3])
x = np.arange(-3, 10, 0.1)
N = len(x)
errors = np.random.normal(0, 0.4, N)
y = mysin(x, p0)
y += errors
model = mysin

# ++++GEN++++
ParamSpace = dict(dimension=2,
                  limits=dict(lower=[0, 0.1], upper=[2, 10]))

# __creating population___
popsize = 12
population = [[np.random.uniform(ParamSpace['limits']['lower'][i], ParamSpace['limits']['upper'][i])
               for i in range(0, ParamSpace['dimension'])] for j in range(0, popsize)]
# np.random.uniform(ParamSpace['limits']['lower'],
#                  ParamSpace['limits']['upper'],
#                  [ParamSpace['dimension'],popsize])
par1 = [par[0] for par in population]
par2 = [par[1] for par in population]


# __breeding__
for i in range(0,popsize-1,2):
    #popmum=population[i]
    #popdad=population[i+1]
    popsun=[population[i+random.randint(0,1)][0],population[i+random.randint(0,1)][1]]
    popdau=[population[i+random.randint(0,1)][0],population[i+random.randint(0,1)][1]]
    population.append(popsun)
    population.append(popdau)

# __calculating residuals__
resids = [residuals(x, y, [par[0], par[1]], mysin) for par in population]
print(population[np.argmin(resids)])

# __selection__
sortorder=np.argsort(resids)[0:popsize]

print(population[sortorder][:])
#newpopulation=population[sortorder]

par1_2 = [par[0] for par in population]
par2_2 = [par[1] for par in population]

# ++++PLOT++++
plt.figure(1)
plt.subplot(221)
plt.plot(x, model(x, p0), 'r', x, y, 'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(1)
plt.title('Data')

plt.subplot(222)
plt.hist(errors)
plt.xlabel('$\Delta y$')
plt.ylabel('$N_{\Delta y}$')
plt.title('Error histogram')

plt.subplot(223)
plt.plot(p0[0],p0[1],'or',par1, par2, 'bo',par1_2,par2_2,'y.')
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.grid(1)
plt.xlabel('Amplitude')
plt.ylabel('$\omega$')
plt.title('Generations')

plt.show()
# plt.show()
