import pygen_module
import numpy as np
import matplotlib.pyplot as plt
import sys

def x2py2(*params):
    if isinstance(params[0], np.ndarray) or isinstance(params[0], list):
        p = params[0]
        # print('array!')
    else:
        p = params

    return p[0]**2+p[1]**2

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

print(sys.path)
p0 = np.array([2.5, 1.3])
x = np.arange(-3, 10, 0.1)
y=0
N = len(x)
errors = np.random.normal(0, 3.4, N)
#model = mysin
#y = model(x, p0)
#y += errors

model=dict(
    type='min',
    fun=x2py2
)
popsize = 6

p0_grid = np.linspace(0, 6, 100)
p1_grid = np.linspace(0, 3, 100)

res_grid = np.zeros([100, 100])
for i in range(0, 100):
    for j in range(0, 100):
        res_grid[i, j] = pygen_module.residuals(x, y, [p0_grid[j], p1_grid[i]], model=model)

param_space = dict(dimension=2,
                   limits=dict(lower=[-2, -2], upper=[2, 2]))

result_lsq, cov, result_gen, bestfit_stack, population_stack = pygen_module.pygenfun(
    x, y, 0, model, param_space, popsize, breeding_model='2p')

# ++++PLOT++++
print(population_stack)
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

    param2plot0 = [pop['parameter'][0] for pop in population_stack[i]]
    param2plot1 = [pop['parameter'][1] for pop in population_stack[i]]
    ax.plot(param2plot0, param2plot1, '.g')
    ax.plot(*result_gen, '*b')
    ax.plot(*result_lsq, '+b')
    ax.grid(1)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('$\omega$')
    ax.set_title(i)
    ax.pcolorfast(p0_grid, p1_grid, res_grid, cmap='Greys')

plt.show()
