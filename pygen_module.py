import numpy as np
import scipy.optimize as opt
# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
from lmfit import minimize, Parameters

def mysin(x, *params):
    if isinstance(params[0], np.ndarray) or isinstance(params[0], list):
        p = params[0]
    else:
        p = params
    return p[0] * np.sin(p[1] * x)


def residuals(data_x: np.ndarray, data_y: np.ndarray,
              params: np.iterable,
              model: callable or dict = mysin,
              errors: np.ndarray = None,
              *args, **kwargs) -> np.ndarray:
    if type(model) == dict:
        try:
            if model['type'] == 'min':
                return model['fun'](*params, **kwargs)
        except Exception:
            pass
        try:
            if model['type'] == 'resids':
                chi = np.sum(np.power(model['fun'](*params, **kwargs), 2))
                return chi
        except Exception:
            pass
    else:
        print("param: ", params)
        if errors is None:
            return np.sum((data_y - model(data_x, *params, **kwargs)) ** 2)
        else:
            errors_nan = np.isnan(errors)
            print(errors_nan)
            return np.nansum(((data_y - model(data_x, *params, **kwargs)) / errors) ** 2)


def getResid(pop):
    return pop['resid']


def selection(population: list, kw_string='no', *args) -> list:
    if kw_string == 'no':
        # no selection
        return population
    elif kw_string == 'ps':
        # to popsize=args[0]
        return population[0:args[0]]


def upgrade_pop(pop, data_x, data_y, model, **kwargs):
    pop['resid'] = residuals(data_x, data_y, pop['parameter'], model, **kwargs)
    print(pop['resid'])
    return pop


def breeding(population: list, kw_string: str,
             data_x, data_y, model,
             breed_popsize=4, number_of_threads=1, **kwargs) -> list:
    """
    Breeds the population for genetic algorithm and calculates new residuals

    :param population:
        list of pops:
        pop: dict([list of parameters]:parameter, float:resid)
    :param kw_string:
        type of breeding:
        'p' for Pairs
        'e' for ~exp(-residuals^2) chance to breed
        largest digit -- breeding parameter
    :param data_x: data to calculate residuals
    :param data_y: data to calculate residuals
    :param model: model to calculate residuals
    :return: new population
    """

    breeding_keywords = {kw for kw in kw_string}
    nchild = max(([int(i) for i in breeding_keywords if i.isdigit()]))
    popsize = len(population)
    nparam = len(population[0]['parameter'])

    if 'p' in breeding_keywords:
        # Pairs:
        # couple pops (0+1, 2+3...) and make N children
        # around hypercube vertices
        # N -- largest digit in kw_string
        popchild = []
        for i in range(0, popsize - 1, 2):
            siblings = [dict(
                parameter=[population[i + np.random.randint(0, 1)]['parameter'][par] * np.random.normal(1, 0.15)
                           for par in range(nparam)],
                resid=0)
                for j in range(0, nchild)]
            popchild.extend(siblings)
        print(len(popchild))

        print("Here!")
        pool = Pool(number_of_threads)
        pops_with_data_and_model = [(pop, data_x, data_y, model) for pop in popchild]
        newpopchild = pool.starmap(upgrade_pop, pops_with_data_and_model)
        """for pop in popchild:
            pop['resid'] = residuals(data_x, data_y, pop['parameter'], model, **kwargs)
        population.extend(popchild)
        """
        population.extend((newpopchild))

    elif 'e' in breeding_keywords:
        # Exponential chance to pair
        # makes approximately kwrargs['children'] or N
        # new pops, chance for every pop to breed is
        # ~a**(-(resi/res0)**2)
        popsize = len(population)
        nparam = len(population[0]['parameter'])
        resid_array = np.array([pop['resid'] for pop in population])
        best_resid = np.min(resid_array)
        a = opt.broyden1(
            lambda a: np.sum(np.power(a, 1 - (resid_array / best_resid))) - breed_popsize,
            2, x_tol=0.5)
        chance = np.power(a, 1 - (resid_array / best_resid))
        print('chances: ', chance)
        breeding_population = [population[i] for i in range(popsize) if np.random.rand() < chance[i]]
        breed_popsize = len(breeding_population)
        print('breed coefficient', a, np.sum(chance))
        for i in range(0, breed_popsize - 1, 2):
            popchild = [dict(
                parameter=[
                    breeding_population[i + np.random.randint(0, 1)]['parameter'][par] * np.random.normal(1, 0.15)
                    for par in range(nparam)],
                resid=0)
                for j in range(0, nchild)]
            for pop in popchild:
                pop['resid'] = residuals(data_x, data_y, pop['parameter'], model, **kwargs)
            population.extend(popchild)
    elif 'h' in breeding_keywords:
        # Hyperbolic chance to pair
        # makes approximately kwrargs['children'] or N
        # new pops, chance for every pop to breed is
        # ~(res0+a)/(resi+a)
        popsize = len(population)
        nparam = len(population[0]['parameter'])
        resid_array = np.array([pop['resid'] for pop in population])
        best_resid = np.min(resid_array)
        print('testing')
        print(breed_popsize)
        print(2. / np.pi * np.arctan(1 / ((resid_array - best_resid) / best_resid)))
        a = opt.broyden1(
            lambda a: 2. / np.pi * np.sum(
                np.arctan(1 / ((resid_array - best_resid) * a / best_resid)))
                      - min(breed_popsize, 0.8 * popsize),
            1, x_tol=0.5, verbose=0)
        chance = np.arctan(1 / ((resid_array - best_resid) * a / best_resid))
        print('chances: ', chance)
        breeding_population = [population[i] for i in range(popsize) if np.random.rand() < chance[i]]
        breed_popsize = len(breeding_population)
        print('breed coefficient', a, np.sum(chance), breed_popsize)
        for i in range(0, breed_popsize - 1, 2):
            popchild = [dict(
                parameter=[
                    breeding_population[i + np.random.randint(0, 1)]['parameter'][par] * np.random.normal(1, 0.15)
                    for par in range(nparam)],
                resid=0)
                for j in range(0, nchild)]
            for pop in popchild:
                pop['resid'] = residuals(data_x, data_y, pop['parameter'], model, **kwargs)
            population.extend(popchild)

    return population


# __________________________________________________________#

def pygenfun(data_x, data_y, y_error,
             model, param_space,
             popsize=10, breeding_model='2p',
             selection_model='ps',
             final_lsq='lm', number_of_threads = 1,
             *args, **kwargs):

    p0 = np.array([2.5, 1.3])
    x = data_x
    y = data_y
    # ++++GEN++++
    population_stack = []
    bestfit_stack = []

    # __creating population___

    # popsize = 12
    nchild = 2

    population = [dict(
        parameter=[np.random.uniform(param_space['limits']['lower'][i],
                                     param_space['limits']['upper'][i])
                   for i in range(0, param_space['dimension'])],
        resid=0)
        for j in range(popsize)]
    # __calculating residuals__

    #for pop in population:
    #    pop['resid'] = residuals(x, y, pop['parameter'], model, **kwargs)
    pool = Pool(number_of_threads)
    pops_with_data_and_model = [(pop, data_x, data_y, model) for pop in population]
    population = pool.starmap(upgrade_pop, pops_with_data_and_model)


    bestfit_stack.append(sorted(population, key=getResid)[0]['parameter'])
    # print(bestfit_stack)
    inprogress = 5
    while inprogress:
        print('STAGE: ', inprogress)
        population_stack.append(population)
        # for pop in population_stack:
        #    print(len(pop), pop)
        # __breeding__
        print('start breeding')
        print(len(population))
        breed_population = breeding(population, breeding_model,
                                    x, y, model, breed_popsize=popsize,
                                    number_of_threads = number_of_threads, **kwargs)
        print(len(population))
        print(len(breed_population))
        # __selection__
        print('selection')
        sorted_population = sorted(breed_population, key=getResid)
        new_population = selection(sorted_population, selection_model, popsize)
        bestfit_stack.append(new_population[0]['parameter'])
        # __shuffle__
        print('shuffle')
        np.random.shuffle(new_population)
        population = new_population
        inprogress -= 0.5

    result_gen = bestfit_stack[-1]

    # print('Initial parameters: ', p0, residuals(x, y, p0))
    print('PyGen parameters: ', result_gen, residuals(x, y, result_gen, model, **kwargs))

    # LM optimization
    print('Curve fitting')
    try:
        if type(model) == dict:
            if model['type'] == 'min':
                print("NOT IMPLEMENTED!")
                quit("Not yet implemented")
                result_lsq_obj = opt.least_squares(model['fun'], result_gen, method=final_lsq)
                result_lsq = result_lsq_obj['x']
                cov = result_lsq_obj['hess_inv']
            if model['type'] == 'resids':
                # parameters_list=tuple(('par'+str(i), result_gen[i]) for i in range(len(result_gen)))
                # fancy_parameters=Parameters()
                ##map(fancy_parameters.add, parameters_list)
                # fancy_parameters.add_many(*parameters_list)
                # result_lsq_obj = minimize(model['fun'], fancy_parameters)
                # result_lsq, cov = result_lsq_obj.params.valuesdict().values, result_lsq_obj.covar

                # result_lsq_obj = opt.leastsq(model['fun'], result_gen, full_output=True)
                result_lsq_obj = opt.least_squares(model['fun'], np.array(result_gen), method='lm', verbose=2)
                print(type(result_lsq_obj))
                print(result_lsq_obj.values)
                resut_lsq = result_lsq_obj[0]
                cov = result_lsq_obj[1]
                print(result_lsq_obj)
                a = opt.OptimizeResult
                print(a.x)
                print(a.hess_inv)
        else:
            result_lsq, cov = opt.curve_fit(model, x, y, result_gen, method=final_lsq, verbose=1, **kwargs)

        print('Least-squares parameters: ', result_lsq, '\u00B1',
              [np.sqrt(np.diag(cov))],
              residuals(x, y, result_lsq, model, **kwargs))

        print(cov)
    except:
        print('WARNING! Least-squares crashed!')
        print('Use gen parameters instead')
        print('(may slow the mcmc process down)')
        result_lsq = result_gen
        cov = np.eye(len(result_gen))
    return result_lsq, cov, result_gen, bestfit_stack, population_stack

    """
    import lmfit
    lm_model=lmfit.Model(model,**kwargs)
    print ('_______')
    print(model)
    print(lm_model.param_names)
    lm_input_params=lmfit.Parameters()
    lm_input_params.add('a',value=result_gen[0])
    lm_input_params.add('b',value=result_gen[1])
    lm_input_params.add('c',value=result_gen[2])
    lm_input_params.add('d',value=result_gen[3])
    lm_input_params.add('e',value=result_gen[4])
    result_lm=lm_model.fit(y,lm_input_params,x=x,verbose=True)
    result_lsq=result_lm.params
    cov=None

    """
    return result_lsq, cov, result_gen, bestfit_stack, population_stack
