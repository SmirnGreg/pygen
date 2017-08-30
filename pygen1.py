import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.optimize as opt


def mysin(x, *params):
    # print('meow')
    # print(params)
    # print(type(params))
    if isinstance(params[0], np.ndarray) or isinstance(params[0], list):
        p = params[0]
        # print('array!')
    else:
        p = params
    # print(p)
    return p[0] * np.sin(p[1] * x)


def mygaus(x, p):
    """
Computes Gaussian
    :param x: np.array or scalar
    :param p: list with at least 2 elements
    :return: 3 * np.exp(-(x - p[0]) ** 2 / (2 * p[1] ** 2))
    """
    return 3 * np.exp(-(x - p[0]) ** 2 / (2 * p[1] ** 2))


def residuals(datax: np.ndarray, datay: np.ndarray,
              params: object,
              model: callable = mysin,
              errors: np.ndarray = np.array([np.nan])) -> np.ndarray:
    """I work!"""
    #return np.sum((datay - model(datax, *params)) ** 2)

    if (errors.shape != datax.shape):
        return np.sum((datay - model(datax, *params)) ** 2)
    else:
        errors_nan = np.isnan(errors)
        print(errors_nan)
        return np.nansum(((datay - model(datax, *params)) / errors) ** 2)


def getResid(pop):
    return pop['resid']


# Defining initial data
# np.random.seed(100564)
p0 = np.array([2.5, 1.3])
x = np.arange(-3, 10, 0.1)
N = len(x)
errors = np.random.normal(0, 3.4, N)
model = mysin
y = model(x, p0)
y += errors

p0_grid = np.linspace(0, 6, 100)
p1_grid = np.linspace(0, 3, 100)

res_grid = np.zeros([100, 100])
for i in range(0, 100):
    for j in range(0, 100):
        res_grid[i, j] = residuals(x, y, [p0_grid[j], p1_grid[i]], model=model)

# ++++GEN++++
population_stack = []
bestfit_stack = []
ParamSpace = dict(dimension=2,
                  limits=dict(lower=[0, 0.1], upper=[6, 3]))

# __creating population___
popsize = 12
nchild = 2
population = [dict(
    parameter=[np.random.uniform(ParamSpace['limits']['lower'][i],
                                 ParamSpace['limits']['upper'][i])
               for i in range(0, ParamSpace['dimension'])],
    resid=0)
    for j in range(0, popsize)]
# __calculating residuals__
for pop in population:
    pop['resid'] = residuals(x, y, pop['parameter'], model)

bestfit_stack.append(sorted(population, key=getResid)[0]['parameter'])
# print(bestfit_stack)
inprogress = 5
while inprogress:
    print('STAGE: ', inprogress)
    population_stack.append(population)
    # __breeding__
    print('start breeding')

    for i in range(0, popsize - 1, 2):
        # popmum=population[i]
        # popdad=population[i+1]
        popchild = [dict(
            parameter=[population[i + np.random.randint(0, 1)]['parameter'][par] * np.random.normal(1, 0.15) for par in
                       range(0, ParamSpace['dimension'])],
            resid=0)
            for j in range(0, nchild)]
        for pop in popchild:
            pop['resid'] = residuals(x, y, pop['parameter'], model)
        population.extend(popchild)
    # __selection__
    print('selection')
    newpopulation = sorted(population, key=getResid)[0:popsize]
    bestfit_stack.append(newpopulation[0]['parameter'])
    # for pop in newpopulation: print(pop)
    np.random.shuffle(newpopulation)
    print('shuffle')
    # for pop in newpopulation: print(pop)
    population = newpopulation
    inprogress -= 0.5

result_gen = bestfit_stack[-1]

print('Initial parameters: ', p0, residuals(x, y, p0))
print('PyGen parameters: ', result_gen, residuals(x, y, result_gen))

# LM optimization
result_lsq, cov = opt.curve_fit(model, x, y, result_gen, method='lm')
print('Least-squares parameters: ', result_lsq, '\u00B1', [np.sqrt(cov[i, i]) for i in range(ParamSpace['dimension'])],
      residuals(x, y, result_lsq))
print(cov)
# ++++PLOT++++
gen2plot = [0, 1, 5]
plt.figure(1)
plt.subplot(221)
plt.plot(x, model(x, p0), 'r', x, y, 'b.')

plt.plot(x, model(x, bestfit_stack[gen2plot[0]]), 'g-')
plt.plot(x, model(x, bestfit_stack[gen2plot[1]]), 'y-')
plt.plot(x, model(x, bestfit_stack[gen2plot[2]]), 'm--')

plt.xlabel('x')
plt.ylabel('y')
plt.grid(1)
plt.title('Data')

plt.subplot(222)
plt.hist(errors)
plt.xlabel('$\Delta y$')
plt.ylabel('$N_{\Delta y}$')
plt.title('Error histogram')

ax = plt.subplot(212)
ax.plot(p0[0], p0[1], 'or')
gen_p0 = [[population_stack[generation][pop]['parameter'][0] for pop in range(0, popsize)] for generation in gen2plot]
gen_p1 = [[population_stack[generation][pop]['parameter'][1] for pop in range(0, popsize)] for generation in gen2plot]
ax.plot(gen_p0[0], gen_p1[0], 'og')
ax.plot(gen_p0[1], gen_p1[1], '.y')
ax.plot(gen_p0[2], gen_p1[2], '*m')
ax.plot(*result_gen, '*b')
ax.plot(*result_lsq, '+b')

plt.subplots_adjust(hspace=0.5, wspace=0.5)
ax.grid(1)
ax.set_xlabel('Amplitude')
ax.set_ylabel('$\omega$')
ax.set_title('Generations')
im = ax.pcolorfast(p0_grid, p1_grid, res_grid, cmap='Greys')
plt.colorbar(im)
plt.figure(2)

for i in range(0, 8):
    # print(i)
    ax = plt.subplot(3, 3, i + 1, sharex=ax, sharey=ax)
    ax.plot(p0[0], p0[1], 'or')

    param2plot0 = [population_stack[i][pop]['parameter'][0] for pop in range(0, popsize)]
    param2plot1 = [population_stack[i][pop]['parameter'][1] for pop in range(0, popsize)]
    ax.plot(param2plot0, param2plot1, '.g')
    ax.plot(*result_gen, '*b')
    ax.plot(*result_lsq, '+b')
    ax.grid(1)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('$\omega$')
    ax.set_title(i)
    ax.pcolorfast(p0_grid, p1_grid, res_grid, cmap='Greys')

plt.show()
# plt.show()
