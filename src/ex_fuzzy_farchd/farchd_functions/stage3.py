#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:51:00 2023

@author: asier
"""
from copy import deepcopy
import numpy as np
import ex_fuzzy.fuzzy_sets as fs
from src.ex_fuzzy_farchd.farchd_functions.Item import items_to_master_rule_base


def translate_ex_fuzzy_partitions(partitions: list, tunning_value_list: list[list[float]]):
    partitionscopy = deepcopy(partitions)
    ix = 0
    for fuzzy_variable in partitionscopy:
        if type(fuzzy_variable[0]) == fs.categoricalFS or type(fuzzy_variable[0]) == fs.categoricalIVFS:
            continue
        domain = fuzzy_variable.domain()
        domain_min, domain_max = domain[0], domain[1]

        nfs = len(fuzzy_variable.linguistic_variables)
        for linguisticc_variable, tuning_value_term in zip(fuzzy_variable.linguistic_variables, tunning_value_list[ix:ix+nfs]):  # var is fuzzySet # FIXME:Categorical!!
            # FIXME: Deal with IVFS
            if fuzzy_variable[0].type() == fs.FUZZY_SETS.t1:
                current_params = np.array(linguisticc_variable.membership_parameters)
                new_params = np.clip(current_params + tuning_value_term, domain_min, domain_max)
                linguisticc_variable.membership_parameters = new_params.tolist()
            elif fuzzy_variable[0].type() == fs.FUZZY_SETS.t2:
                current_lower = linguisticc_variable.secondMF_lower
                current_upper = linguisticc_variable.secondMF_upper
                new_lower = np.clip(current_lower + tuning_value_term, domain_min, domain_max)
                new_upper = np.clip(current_upper + tuning_value_term, domain_min, domain_max)
                linguisticc_variable.secondMF_lower = new_lower.tolist()
                linguisticc_variable.secondMF_upper = new_upper.tolist()

    return partitionscopy


def real_to_gray(number, bits=30):
    scaled = np.uint32((2**30-1) * (number + 0.5) + 0.5)
    gray = np.bitwise_xor(scaled >> 1, scaled)
    gray = np.binary_repr(gray, bits)
    gray = np.array([int(n) for n in gray])
    return gray

def reshape_chromosome(chromosome_flat, master_rule_base):
    lengths = [len(sublist) for sublist in master_rule_base]
    reshaped = []
    idx = 0
    for l in lengths:
        reshaped.append(chromosome_flat[idx:idx+l])
        idx += l
    return reshaped

def rule_base_from_chromosome(master_rule_base, chromosome, classes_dict, partitions):
    if sum(chromosome[0]) == 0:
        # FIXME: return empty rule base?
        return 0
    new_rule_base = items_to_master_rule_base(master_rule_base, classes_dict, partitions)
    rules_activations = reshape_chromosome(chromosome[0], new_rule_base)

    for rule_base, activations in zip(new_rule_base, rules_activations):
        for ix, (rule, active) in enumerate(zip(rule_base, activations)):
            if active == 0:
                rule_base.remove_rule(ix)

    translated_partitons = translate_ex_fuzzy_partitions(partitions, chromosome[1])
    new_rule_base.antecedents = translated_partitons

    return new_rule_base

def chromosome_evaluation_exfuzzy(chromosome, master_rule_base, partitions, classes_dict, x, y, delta):
    N = len(y)
    Hits = 0
    if (active_rules := sum(chromosome[0])) == 0:
        return 0
    new_rule_base = rule_base_from_chromosome(master_rule_base, chromosome, classes_dict, partitions)

    y_out = new_rule_base.predict(x)
    Hits += sum(y_out == y.map(classes_dict))

    NRinitial = len(master_rule_base)  # number of candidate rules
    NR = active_rules  # number of selected rules
    Fitness = Hits/N - delta * NRinitial/(NRinitial - NR + 1.0)

    return Fitness


def gray_distance(chromosome1, chromosome2, BITSGENE):
    c1, c2 = chromosome1[0], chromosome2[0]
    c1i = np.array(list(map(real_to_gray, chromosome1[1])))
    c2i = np.array(list(map(real_to_gray, chromosome2[1])))
    return np.sum(c1 != c2) + np.sum(c1i != c2i)


def crossover(parent1, parent2, alpha=1):
    diff_cs = parent1[0] != parent2[0]
    diff_idxs = np.where(diff_cs)
    n_interchange = sum(diff_cs)//2
    selected_idx = np.random.choice(diff_idxs[0], size=n_interchange,
                                    replace=False)
    offspring1 = [parent1[0].copy(), parent1[1].copy()]
    offspring2 = [parent2[0].copy(), parent2[1].copy()]

    # HUX
    offspring1[0][selected_idx] = parent2[0][selected_idx]
    offspring2[0][selected_idx] = parent1[0][selected_idx]

    # PCBLX
    ai = np.minimum(offspring1[1], offspring2[1])
    bi = np.maximum(offspring1[1], offspring2[1])
    alpha_I = alpha * abs(ai-bi)
    li1 = np.maximum(ai, offspring1[1]-alpha_I)
    Ui1 = np.minimum(bi, offspring1[1]+alpha_I)
    li2 = np.maximum(ai, offspring2[1]-alpha_I)
    Ui2 = np.minimum(bi, offspring2[1]+alpha_I)
    offspring1[1] = np.random.uniform(li1, Ui1)
    offspring2[1] = np.random.uniform(li2, Ui2)

    return offspring1, offspring2


def start_population(rules, partitions, x, y, classes_dict, pop, delta):
    CS = np.ones(len(rules))
    CT = np.zeros(sum([len(partition.linguistic_variables) for partition in partitions if type(partition[0]) != fs.categoricalFS]))
    population = [[CS, CT]]
    fitnesses = [chromosome_evaluation_exfuzzy(population[-1], rules,
                                       partitions, classes_dict,x, y, delta)]
    for _ in range(pop-1):
        CSi = np.random.randint(2, size=len(rules))
        CTi = np.random.uniform(low=-0.5, high=0.5, size=len(CT))
        population.append([CSi, CTi])
        fitnesses.append(chromosome_evaluation_exfuzzy(population[-1], rules,
                                               partitions, classes_dict,x, y, delta))
    return population, fitnesses


# Stage 3. Rule Selection and Lateral Tuning
def stage3(rules, x, y, partitions, classes_dict, pop: int, evaluations: int, BITSGENE: int, delta: float):
    # 12. Genereate the initial population with P chromosomes
    # 13. Evaluate the population
    # # 14. Initialize the threshold value taking into account Gray codings,
    # # i.e., L=Linitial
    # Linitial =  max_dist/4
    # L = Linitial

    population, fitnesses = start_population(rules, partitions,
                                             x, y, classes_dict, pop, delta)

    lenCT = len(population[0][1])
    Linitial = len(rules[0].variables+lenCT*BITSGENE)/4
    L = Linitial
    n_eval = 1
    # 15. Generate the next population as following.
    while n_eval < evaluations:  # FIXME: names
        # 1) Shuffle the populations
        p = np.random.permutation(len(population)-1)
        population = np.array(population, dtype=object)[p]
        fitnesses = np.array(fitnesses)[p]
        best_fitness_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_fitness_idx]
        best_chromosome = population[best_fitness_idx]
        # 2)Select the parents two by two. Each pair is crossed if the hamming
        # distance between the parent Gray codings divided by 2 is more than L
        offspring = []
        offspring_eval = []
        for i in range(0, len(population)//2, 2):
            dist = gray_distance(population[i], population[i+1], BITSGENE)
            if dist/2 > L:
                offspring1, offspring2 = crossover(population[i],
                                                   population[i+1])
                child1_eval = chromosome_evaluation_exfuzzy(offspring1, rules,
                                                    partitions, classes_dict, x, y, delta)
                child2_eval = chromosome_evaluation_exfuzzy(offspring2, rules,
                                                    partitions, classes_dict,x, y, delta)
                offspring.append(offspring1)
                offspring.append(offspring2)
                # 3) Evaluate the new individuales
                offspring_eval.append(child1_eval)
                offspring_eval.append(child2_eval)
        n_eval += 1
        # 4) Join the parents with their offspring, and select the best P
        # individuals to take part in the next population
        population = np.vstack([population, np.array(offspring,dtype=object)])
        fitnesses = np.hstack([fitnesses, offspring_eval])
        fitnesses, population = map(list,
                                    zip(*sorted(zip(fitnesses, population),
                                                key=lambda x: x[0],
                                                reverse=True)))
        population = population[:pop]
        fitnesses = fitnesses[:pop]
        # 16. If the best chromosome does not change or there are no
        # new individuals in the population, then L= L - (Linitial * 0,1)
        if np.all(best_chromosome[0] == population[0][0]) and \
           np.all(best_chromosome[1] == population[0][1]):
            # FIXME: DeprecationWarning: elementwise comparison failed;
            # this will raise an error in the future.
            #  if best_chromosome == population[0]:
            L = L - Linitial*0.1
        # 17. If L <0, restart the population and initialize L
        # The best chromosome is maintained and the remaining are generated
        # at random
        if L < 0:
            population, fitnesses = start_population(rules, partitions,
                                                     x, y, classes_dict, pop, delta)
            population[0] = best_chromosome
            fitnesses[0] = best_fitness
            L = Linitial
        # 18. If the maximum number of evaluations isn't reached, go to Step 15
    final_rules = rule_base_from_chromosome(rules, population[0], classes_dict, partitions)
    return final_rules
