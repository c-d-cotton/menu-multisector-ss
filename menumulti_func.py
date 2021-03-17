#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import copy
import datetime
import functools
import multiprocessing
import numpy as np
import pickle
from scipy.optimize import brentq
import shutil

import matplotlib.pyplot as plt

# Defaults:{{{1
sys.path.append(str(__projectdir__ / Path('submodules/menu-sim-general/')))
from generalmenu_func import starttime
starttime = starttime
sys.path.append(str(__projectdir__ / Path('submodules/menu-sim-general/')))
from generalmenu_func import SIGMA
SIGMA = SIGMA
numsectors_default = 9
sys.path.append(str(__projectdir__ / Path('submodules/menu-sim-general/')))
from generalmenu_func import keydetails
keydetails = keydetails
multikeydetails = ['weights', 'multicesrelprice', 'NUglobal', 'MCglobal', 'aggMCglobal', 'profitshareglobal', 'menushareglobal', 'profitshareglobalmenu']

# Paramssdict:{{{1
def getadjustedparamssdict(p = None):
    if p is None:
        p = {}

    # turn off print for individual sectors because that would get messy
    if 'printinfosummary' not in p:
        p['printinfosummary'] = False

    # DECIDED TO CANCEL THIS
    # unless specify parameters, use lower number of states since I have lots of sectors so I don't need to be as precise in individual sectors
    # and code would take forever with large number of states and many sectors
    if 'num_pi' not in p:
        p['num_pi'] = 200
    if 'num_A' not in p:
        p['num_A'] = 50

    return(p)


def getmultiparamssdict(pmulti = None):
    if pmulti is None:
        pmulti = {}

    pmultidef = {}


    pmultidef['printinfosummary'] = True

    for param in pmultidef:
        if param not in pmulti:
            pmulti[param] = pmultidef[param]

    # ADD WEIGHTS, FREQUENCIES
    if 'numsectors' not in pmulti:
        if 'weights' in pmulti:
            pmulti['numsectors'] = len(pmulti['weights'])
        else:
            pmulti['numsectors'] = numsectors_default

    # menu costs:{{{
    # solved for 9 sector NS framework, MC was 0.880328
    menusdef = [7.494e-5, 0.00747, 0.01291, 0.01816, 0.03096, 0.04539, 0.05820, 0.0608, 0.09329]
    menufile = __projectdir__ / Path('temp/calib_basic.txt')
    if 'menus' in pmulti:
        None
    elif 'menusdef' in pmulti and pmulti['menusdef'] is True:
        pmulti['menus'] = menusdef
    elif os.path.isfile(menufile):
        with open(menufile) as f:
            menus = f.read()
        if menus[-1] == '\n':
            menus = menus[: -1]
        pmulti['menus'] = [float(menu) for menu in menus[1: -1].split(',')]
    else:
        if pmulti['numsectors'] == len(menusdef):
            print('Warning: menus not specified so used default.')
            p['menus'] = menusdef
        else:
            raise ValueError('Need to specify menus.')
    # menu costs:}}}

    if 'weights' not in pmulti:
        sys.path.append(str(__projectdir__ / Path('submodules/calvo-multisector-ss/')))
        from manysector_ss_func import ns_vectors
        pmulti['weights'], pmulti['pricechangeprobs'] = ns_vectors(numsectors = pmulti['numsectors'])

    # verify all same length
    if len(pmulti['weights']) != pmulti['numsectors']:
        raise ValueError('len weights != numsectors')

    if np.sum(pmulti['weights']) != 1:
        raise ValueError('weights not sum to 1')


    return(pmulti)


# Solve partial:{{{1
def multipartialeq_solve_p(pmulti = None, p = None):
    p = getadjustedparamssdict(p = p)

    pmulti = getmultiparamssdict(pmulti = pmulti)

    pmulti['ps'] = []
    for i in range(pmulti['numsectors']):
        p2 = copy.deepcopy(p)
        p2['menu'] = pmulti['menus'][i]
        sys.path.append(str(__projectdir__ / Path('submodules/menu-sim-general/')))
        from generalmenu_func import partialeq_solve_p
        p2 = partialeq_solve_p(p2)
        pmulti['ps'].append(p2)

    pmulti['MCglobal'] = pmulti['ps'][0]['MC']
    pmulti['multicesrelprice'] = np.sum([pmulti['weights'][i] * pmulti['ps'][i]['cesrelprice'] ** (1 - pmulti['ps'][0]['SIGMA']) for i in range(pmulti['numsectors'])])
    pmulti['NUglobal'] = np.sum([pmulti['weights'][i] * pmulti['ps'][i]['NU'] for i in range(pmulti['numsectors'])])
    pmulti['aggMCglobal'] = pmulti['MCglobal'] * pmulti['NUglobal']
    pmulti['profitshareglobal'] = 1 - pmulti['aggMCglobal']

    if pmulti['ps'][0]['pol_details'] is True:
        pmulti['menushareglobal'] = np.sum([pmulti['weights'][i] * pmulti['menus'][i] * pmulti['ps'][i]['price_change_prob'] for i in range(pmulti['numsectors'])])
        pmulti['profitshareglobalmenu'] = pmulti['profitshareglobal'] - pmulti['menushareglobal']



    if pmulti['printinfosummary'] is True:
        # first print all individual items
        for item in sorted(keydetails):
            if item in pmulti['ps'][0]:
                multilist = [pmulti['ps'][i][item] for i in range(pmulti['numsectors'])]
                print(item + ': ' + str(multilist))

        # now print aggregate items
        for item in sorted(multikeydetails):
            if item in pmulti:
                print(item + ': ' + str(pmulti[item]))


    return(pmulti)


def getcalib():
    multipartialeq_solve_p(p = {'MC': 0.8803, 'pistar': 0.02562})


def multipartialeq_solve_p_test():
    pmulti = {}
    pmulti['weights'] = [0.4, 0.6]
    pmulti['menus'] = [0.03, 0.04]

    p = {}
    p['doquick'] = True

    multipartialeq_solve_p(pmulti, p)


# Solve MC:{{{1
def multigeneraleq_aux_p(pmulti, p, MC):
    """
    Auxilliary function used when do generaleq_solve
    """
    # need to make copy every time I run function to ensure I don't end up running on same p
    p = copy.deepcopy(p)
    if p is None:
        p = {}
    p['MC'] = MC
    # no point in solving for policy function details
    p['pol_details'] = False
    multipartialeq_solve_p(pmulti, p)
    if pmulti['printinfosummary'] is True:
        print('\nMulti MC solve current iteration cesrelprice: ' + str(pmulti['multicesrelprice']) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')
        print('\n')

    return(pmulti['multicesrelprice'] - 1)


def solvemultiMC(pmulti = None, p = None, MClow = None, MChigh = None, tolerance = 1e-6):
    if p is None:
        p = {}

    # want a small range to prevent errors and make code run faster
    # unless I have num_A very small or extreme values, I should always be in this approximate range
    if 'SIGMA' not in p:
        p['SIGMA'] = SIGMA
    if MClow is None:
        MClow = (p['SIGMA'] - 1) / p['SIGMA'] - 0.025
    if MChigh is None:
        MChigh = (p['SIGMA'] - 1) / p['SIGMA'] + 0.025

    f1 = functools.partial(multigeneraleq_aux_p, pmulti, p)

    p['MC'] = brentq(f1, MClow, MChigh, xtol = tolerance)

    # get pol details
    p['pol_details'] = True
    # add rest of solution
    pmulti = multipartialeq_solve_p(pmulti, p)

    if pmulti['printinfosummary'] is True:
        print('\n\nSOLUTION FOR MC: ' + str(p['MC']) + '. Time: ' + str(datetime.datetime.now() - starttime))

    return(p)




def solvemultiMC_test():
    p = {}
    # p['doquick'] = True

    pmulti = {}
    # pmulti['weights'] = [0.4, 0.6]
    # pmulti['menus'] = [0.04153, 0.04153]
    pmulti['weights'] = [1]
    pmulti['menus'] = [0.04153]

    solvemultiMC(pmulti = pmulti, p = p)


# Calibrate Menu:{{{1
def solvemultimenu(pmulti = None, p = None, inflation = 0.02562, menuslow = None, menushigh = None, tolerance = 1e-6):
    if p is None:
        p = {}
    p['printinfosummary'] = True
    p = getadjustedparamssdict(p = p)
    pmulti = getmultiparamssdict(pmulti = pmulti)

    if menuslow is None:
        menuslow = []
        for pricechangeprob in pmulti['pricechangeprobs']:
            if pricechangeprob > 0.8:
                menulow = 0.00001
            elif pricechangeprob > 0.2:
                menulow = 0.0001
            else:
                menulow = 0.001
            menuslow.append(menulow)
    if menushigh is None:
        menushigh = []
        for pricechangeprob in pmulti['pricechangeprobs']:
            if pricechangeprob < 0.05:
                menuhigh = 0.25
            else:
                menuhigh = 0.1
            menushigh.append(menuhigh)

    menus = []
    for i in range(pmulti['numsectors']):
        pricechangeprob = pmulti['pricechangeprobs'][i]

        # need to set a lower menulow to deal with high price change frequency cases
        sys.path.append(str(__projectdir__ / Path('submodules/menu-sim-general/')))
        from generalmenu_func import solvemenu_givenMC
        menu = solvemenu_givenMC(p = copy.deepcopy(p), pricechangeprob = pricechangeprob, inflation = inflation, menulow = menuslow[i], menuhigh = menushigh[i], tolerance = tolerance)
        menus.append(menu)

    if p['printinfosummary'] is True:
        print('\n')
        print('Price change probs: ' + str(pmulti['pricechangeprobs']))
        print('MULTI SOLUTION FOR MENUS: ' + str(menus))
    return(menus)


def solvemultimenu_test():
    p = {'doquick': True}
    solvemultimenu(p = p)


# Calibrate Menu and MC Same Time:{{{1
def solvemultimenuMC(pmulti0 = None, p0 = None, pricechangeprob = 0.087, inflation = 0.02562, printthis = True, MClow_input = None, MChigh_input = None, menuprecision = 2e-4, MCprecision = 2e-4, raiseerrorifexitearly = False, savefile = None):
    """
    solvemenu_givenMC does solvemenu(MC) = menu* where MC is not MC*
    I want to iterate over solveMC(menu) and solvemenu(MC) to find menu*, MC* s.t. solveMC(menu*) = MC* and solvemenu(MC*) = menu*

    To do this I first solve for MClow and MChigh given menulow and menuhigh under MCinit
    I then do the same for menulow and menuhigh
    I keep doing this until I hopefully converge

    Precision shouldn't be too narrow since at small values I think the solve functions could go in the wrong direction.
    Normal directions:
    - When MC rises, solved for menu* goes down. I think this is because for a given menu, with higher MC, firms get lower benefit from changing prices so price change probabilities are lower. Therefore, we need a lower menu cost to get higher price change probabilities.
    - When menu rises, MC goes down. I think this is because with a higher menu, price dispersion is lower so the CES price aggregator is naturally lower. Therefore, MC does not need to rise by as much.
    """
    if p0 is None:
        p0 = {}
    p0['pistar'] = inflation
    if 'SIGMA' not in p0:
        p0['SIGMA'] = SIGMA
    MCinit = (p0['SIGMA'] - 1) / p0['SIGMA']
    if MClow_input is None:
        MClow_input = MCinit - 0.025
    if MChigh_input is None:
        MChigh_input = MCinit + 0.025

    if pmulti0 is None:
        pmulti0 = {}

    solveformenu_thisiteration = True
    iterationi = 0
    MClow_output = MClow_input
    MChigh_output = MChigh_input 
    # initial input range for menu cost - just set to None since then the menu cost code can fill it in better
    menuslow_input = None
    menushigh_input = None
    menuiterationprecision = menuprecision / 5
    MCiterationprecision = MCprecision / 5

    while True:
        # do actual iteration
        if solveformenu_thisiteration is True:
            p = copy.deepcopy(p0)
            pmulti = copy.deepcopy(pmulti0)
            p['MC'] = MClow_input
            # constrain to consider only menu costs between menulow_input and menuhigh_input
            menushigh_output = solvemultimenu(pmulti = pmulti, p = p, menuslow = menuslow_input, menushigh = menushigh_input, tolerance = menuiterationprecision)

            p = copy.deepcopy(p0)
            pmulti = copy.deepcopy(pmulti0)
            p['MC'] = MChigh_input
            menuslow_output = solvemultimenu(pmulti = pmulti, p = p, menuslow = menuslow_input, menushigh = menushigh_input, tolerance = menuiterationprecision)

        else:
            p = copy.deepcopy(p0)
            pmulti = copy.deepcopy(pmulti0)
            pmulti['menus'] = menuslow_input
            p = solvemultiMC(pmulti = pmulti, p = p, MClow = MClow_input, MChigh = MChigh_input, tolerance = MCiterationprecision)
            MChigh_output = p['MC']

            p = copy.deepcopy(p0)
            pmulti = copy.deepcopy(pmulti0)
            pmulti['menus'] = menushigh_input
            p = solvemultiMC(pmulti = pmulti, p = p, MClow = MClow_input, MChigh = MChigh_input, tolerance = MCiterationprecision)
            MClow_output = p['MC']

        # print basic details on iterations
        if pmulti['printinfosummary'] is True:
            print('\n\n\nITERATION COMPLETED: ' + str(iterationi))
            print('Iteration was solving for menu: ' + str(solveformenu_thisiteration))
            print('menuslow_input:   ' + str(menuslow_input))
            print('menushigh_input:  ' + str(menushigh_input))
            print('menuslow_output:  ' + str(menuslow_output))
            print('menushigh_output: ' + str(menushigh_output))
            print('MClow_input:   ' + str(MClow_input))
            print('MChigh_input:  ' + str(MChigh_input))
            print('MClow_output:  ' + str(MClow_output))
            print('MChigh_output: ' + str(MChigh_output))
            print('\n')

        # checks to ensure iterations haven't failed
        # either MC is bad or menuslow_input is not None (so not first iteration) and menus are bad
        if MClow_output > MChigh_output + 2 * MCiterationprecision or MClow_output < MClow_input or MChigh_output > MChigh_input or          (menuslow_input is not None and (True in [menuslow_output[i] > menushigh_output[i] + 2 * menuiterationprecision for i in range(len(menuslow_output))] or True in [menuslow_output[i] < menuslow_input[i] for i in range(len(menuslow_output))] or True in [menushigh_output[i] > menushigh_input[i] for i in range(len(menuslow_output))])):
            if raiseerrorifexitearly is True:
                raise ValueError('Iteration Failed.')
            else:
                print('WARNING: Iteration Failed.')
                badsol = [0.5 * (menuslow_output[i] + menushigh_output[i]) for i in range(len(menuslow_output))]
                print('Best guess for menu cost: ' + str(badsol))
                return(badsol)

        if MChigh_output - MClow_output < MCprecision is True and False not in [menushigh_output[i] - menuslow_output[i] < menuprecision for i in range(len(menushigh_output))]:
            sol = [0.5 * (menuslow_output[i] + menushigh_output[i]) for i in range(len(menuslow_output))]
            break

        # if menulow_output and menuhigh_output are very close then we could get the problem where for a given MC, the correct menu is not in the range of [menulow_output, menuhigh_output]
        # to get around this problem, I widen the range in this case
        for i in range(len(menuslow_output)):
            if menushigh_output[i] - menuslow_output[i] < 2 * menuiterationprecision:
                print('Adjusting menu costs for sector: ' + str(i))
                print('Menulow before: ' + str(menuslow_output[i]) + '. Menuhigh before: ' + str(menushigh_output[i]) + '.')
                menushigh_output[i] = menushigh_output[i] + menuiterationprecision
                menuslow_output[i] = menuslow_output[i] - menuiterationprecision
                print('Menulow after: ' + str(menuslow_output[i]) + '. Menuhigh after: ' + str(menushigh_output[i]) + '.')

        # if MC become very close then it could pose problems for brentq and I may as well just take the average and find the menu costs for this MC
        if MChigh_output - MClow_output < 2 * MCiterationprecision:
            print('\n\nMClow_output and MChigh_output very close so stopping iterations and solving directly for menu costs.\n')
            p = copy.deepcopy(p0)
            pmulti = copy.deepcopy(pmulti0)
            p['MC'] = (MClow_output + MChigh_output) / 2
            sol = solvemultimenu(pmulti = pmulti, p = p, menuslow = menuslow_input, menushigh = menushigh_input, tolerance = menuprecision / 5)
            print('Marginal cost: ' + str(p['MC']))
            break
            

        # solve for other one next iteration
        if solveformenu_thisiteration is True:
            menushigh_input = menushigh_output
            menuslow_input = menuslow_output
        else:
            MChigh_input = MChigh_output
            MClow_input = MClow_output
        solveformenu_thisiteration = not solveformenu_thisiteration
        iterationi += 1

    print('Menu cost solution: ' + str(sol))

    # save results
    if savefile is not None:
        with open(savefile, 'w+') as f:
            f.write(str(sol))

    return(sol)


def solvemultimenuMC_test():
    p0 = {'doquick': True}

    pmulti0 = {}
    pmulti0['weights'] = [0.4, 0.6]
    pmulti0['pricechangeprobs'] = [0.06, 0.1]

    solvemultimenuMC(p0 = p0, pmulti0 = pmulti0, savefile = None)


# Save Function General:{{{1
def savesinglerun(filenamenostem, pmulti, p, skipfileifexists = False):
    if skipfileifexists is True and os.path.isfile(filenamenostem + '.pickle') is True:
        print('Skipped file: ' + filenamenostem + '.')
        return(0)
            
    solvemultiMC(pmulti = pmulti, p = p)

    savedict = {name: pmulti[name] for name in multikeydetails if name in pmulti}
    # add keydetails from p dictionary
    for name in keydetails:
        if name in pmulti['ps'][0]:
            savedict[name] = [pmulti['ps'][i][name] for i in range(len(pmulti['ps']))]

    with open(filenamenostem + '.pickle', 'wb') as f:
        pickle.dump(savedict, f)

    toprint = sorted([name + ': ' + str(savedict[name]) for name in savedict])
    with open(filenamenostem + '.txt', 'w+') as f:
        f.write('\n'.join(toprint))

    
def savesingleparamfolder_aux(folder, pmulti0, p0, paramname, skipfileifexists, param):
    pmulti = copy.deepcopy(pmulti0)
    p = copy.deepcopy(p0)
    p[paramname] = float(param)

    print('Started: ' + paramname + ': ' + str(param) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')

    savesinglerun(folder + str(param), pmulti, p, skipfileifexists = skipfileifexists)

    print('Finished: ' + paramname + ': ' + str(param) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')


def savesingleparamfolder(folder, paramname, params, pmulti0 = None, p0 = None, replacefolder = True, skipfileifexists = False, multiprocessthis = True):
    """
    params should be inputted as a list of strings if I want to ensure I get the correct save format i.e. to prevent str(0.02)=0.19999999999
    """
    if p0 is None:
        p0 = {}
    if pmulti0 is None:
        pmulti0 = {}
    if multiprocessthis is True:
        # would be very messy if I had multiprocessing with printinfosummary on
        pmulti0['printinfosummary'] = False

    if replacefolder is True and os.path.isdir(folder):
        shutil.rmtree(folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # get auxilliary function
    f1 = functools.partial(savesingleparamfolder_aux, folder, pmulti0, p0, paramname, skipfileifexists)

    if multiprocessthis is True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(f1, params)
    else:
        for i in range(len(params)):
            f1(params[i])


def savepistars(multiprocessthis = False, replacefolder = True, skipfileifexists = False, pmulti0 = None, p0 = None):
    # do it in this order to get a sense of what the results will be sooner
    pistars = [0, 0.02, 0.04, 0.01, -0.01, -0.005, -0.001, 0.001, 0.005, 0.015, 0.025, 0.03, 0.035]
    savesingleparamfolder(__projectdir__ / Path('temp/pistars/'), 'pistar', pistars, replacefolder = replacefolder, skipfileifexists = skipfileifexists, multiprocessthis = multiprocessthis, pmulti0 = pmulti0, p0 = p0)

def savepistars_test(multiprocessthis = False):

    pmulti0 = {}
    pmulti0['weights'] = [0.4, 0.6]
    pmulti0['menus'] = [0.03, 0.04]

    p0 = {'doquick': True}

    savepistars(multiprocessthis = multiprocessthis, pmulti0 = pmulti0, p0 = p0)


# Graphs:{{{1
def graph_pistar_profitshare(show = False):
    sys.path.append(str(__projectdir__ / Path('submodules/menu-sim-general/')))
    from generalmenu_func import loadsingleparamfolder
    pistars, plist = loadsingleparamfolder(__projectdir__ / Path('temp/pistars/'))
    profitshares = [100 * (1 - p['aggMCglobal']) for p in plist]
    profitsharesmenu = [100 * (1 - p['aggMCglobal'] - p['menushareglobal']) for p in plist]

    plt.plot(pistars, profitsharesmenu, label = 'External Menu Costs')
    plt.plot(pistars, profitshares, label = 'Internal Menu Costs')

    plt.xlabel('Trend Inflation')
    plt.ylabel('Profit Share (\%)')
    
    plt.legend()

    plt.savefig(__projectdir__ / Path('temp/graphs/pistar_profitshare.png'))
    if show is True:
        plt.show()

    plt.clf()


def fullgraphs():
    if os.path.isdir(__projectdir__ / Path('temp/graphs/')):
        shutil.rmtree(__projectdir__ / Path('temp/graphs/'))
    os.mkdir(__projectdir__ / Path('temp/graphs/'))
    graph_pistar_profitshare()


# Load Function:{{{1
def interpolatepistar(pistar):
    """
    Read pistars from folder and then interpolate them to get estimates of MC, NU and menushare in that case
    pistar should be in range of pistars in folder
    """
    sys.path.append(str(__projectdir__ / Path('submodules/menu-sim-general/')))
    from generalmenu_func import loadsingleparamfolder
    pistars, retdictlist = loadsingleparamfolder(__projectdir__ / Path('temp/pistars/'))

    # switch to using MCglobal
    MClist = [retdict['MCglobal'] for retdict in retdictlist]
    NUlist = [retdict['NUglobal'] for retdict in retdictlist]
    menusharelist = [retdict['menushareglobal'] for retdict in retdictlist]

    MC = np.interp(pistar, pistars, MClist)
    NU = np.interp(pistar, pistars, NUlist)
    menushare = np.interp(pistar, pistars, menusharelist)

    return(MC, NU, menushare)


# Full:{{{1
def full():
    solvemultimenuMC(savefile = __projectdir__ / Path('temp/calib_basic.txt'))
    savepistars()
