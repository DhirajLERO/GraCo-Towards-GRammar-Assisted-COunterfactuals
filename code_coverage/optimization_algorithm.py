import random
import math
import numpy as np
import time
import warnings
import pandas as pd
import os

from deap import tools
from deap.benchmarks.tools import hypervolume


def pattern(seq):
    storage = {}
    max_freqs = []
    for length in range(5,int(len(seq)/2)+1):
        valid_strings = {}
        for start in range(0,len(seq)-length+1):
            valid_strings[start] = tuple(seq[start:start+length])
        candidates = set(valid_strings.values())
        if len(candidates) != len(valid_strings):
            storage = valid_strings
            freq = []
            for v in storage.values():
                if list(storage.values()).count(v) > 1:
                    freq.append(list(storage.values()).count(v))
            current_max = max(freq)
            max_freqs.append([length, current_max])
        else:
            break
    return max_freqs


def varAnd(population, toolbox, cxpb, mutpb,
           bnf_grammar, codon_size, max_tree_depth, codon_consumption,
           invalidate_max_depth,
           genome_representation, max_genome_length):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    """
    offspring = [toolbox.clone(ind) for ind in population]
    #    invalid = [ind for ind in population if ind.invalid]
    #    print("number of invalids going to cross/mut", len(invalid))

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i],
                                                          bnf_grammar,
                                                          max_tree_depth,
                                                          codon_consumption,
                                                          invalidate_max_depth,
                                                          genome_representation,
                                                          max_genome_length)
            # del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        offspring[i], = toolbox.mutate(offspring[i], mutpb,
                                       codon_size, bnf_grammar,
                                       max_tree_depth, codon_consumption,
                                       invalidate_max_depth,
                                       max_genome_length)
        # del offspring[i].fitness.values

    return offspring


class hofWarning(UserWarning):
    pass


def ge_eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, elite_size,
                           bnf_grammar, codon_size, max_tree_depth,
                           max_genome_length=None,
                           points_train=None, points_test=None, codon_consumption='eager',
                           report_items=None,
                           genome_representation='list',
                           invalidate_max_depth=False,
                           problem=None,
                           stats=None, halloffame=None,
                           final_gen_file_location=None,
                           save_final_gen=False,
                           save_every_20_gen=False,
                           verbose=__debug__,
                           algo = None,
                           MU=None):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_, with some adaptations to run GE
    on GRAPE.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param elite_size: The number of best individuals to be copied to the
                    next generation.
    :params bnf_grammar, codon_size, max_tree_depth: Parameters
                    used to mapper the individuals after crossover and
                    mutation in order to check if they are valid.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """

    logbook = tools.Logbook()
    if report_items:
        logbook.header = report_items
    else:
        logbook.header = ['gen', 'invalid'] + (stats.fields if stats else [])

    if halloffame is None:
        if elite_size != 0:
            raise ValueError("You should add a hof object to use elitism.")
        else:
            warnings.warn('You will not register results of the best individual while not using a hof object.',
                          hofWarning)
            # logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if elite_size > halloffame.maxsize:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to ELITE_SIZE")

    start_gen = time.time()
    evaluated_inds = 0
    # Evaluate the individuals with an invalid fitness
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
            # print(ind.fitness.values)
            evaluated_inds += 1

    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]

    if len(valid0) != len(valid):
        warnings.warn(
            "Warning: There are valid individuals with fitness = NaN in the population. We will avoid them in the statistics and selection process.")
    invalid = len(population) - len(
        valid0)  # We use the original number of invalids in this case, because we just want to count the completely mapped individuals

    list_structures = []

    # for ind in offspring:
    for idx, ind in enumerate(valid):
        list_structures.append(str(ind.structure))

    unique_structures = np.unique(list_structures, return_counts=False)

    structural_diversity = len(unique_structures) / len(population)

    length = [len(ind.genome) for ind in valid]
    avg_length = sum(length) / len(length)

    nodes = [ind.nodes for ind in valid]
    avg_nodes = sum(nodes) / len(nodes)

    depth = [ind.depth for ind in valid]
    avg_depth = sum(depth) / len(depth)

    used_codons = [ind.used_codons for ind in valid]
    avg_used_codons = sum(used_codons) / len(used_codons)

    end_gen = time.time()
    generation_time = end_gen - start_gen

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(valid)
        best_ind_length = len(halloffame.items[0].genome)
        best_ind_nodes = halloffame.items[0].nodes
        best_ind_depth = halloffame.items[0].depth
        best_ind_used_codons = halloffame.items[0].used_codons
        if not verbose:
            print("gen =", 0, ", Fitness =", halloffame.items[0].fitness.values[0], ", Invalids =", invalid)

    if points_test:
        fitness_test = np.NaN

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, invalid=invalid, train_fitness=halloffame.items[0].fitness.values[0],
                   best_ind_length=best_ind_length, avg_length=avg_length,
                   best_ind_nodes=best_ind_nodes,
                   avg_nodes=avg_nodes,
                   best_ind_depth=best_ind_depth,
                   avg_depth=avg_depth,
                   avg_used_codons=avg_used_codons,
                   best_ind_used_codons=best_ind_used_codons,
                   structural_diversity=structural_diversity,
                   evaluated_inds=evaluated_inds,
                   avg_n_cases=0)

    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(logbook.select("gen")[-1] + 1, ngen + 1):
        start_gen = time.time()

        # Select the next generation individuals
        start = time.time()
        print(len(valid), len(population) - elite_size)
        if algo == "weighted":
            offspring = toolbox.select(valid, len(population) - elite_size)
        elif algo == "NSGAII":
            if gen == 1:
                offspring = toolbox.select(valid, len(population) - elite_size)
            else:
                offspring = toolbox.select(valid + old_population, len(population) - elite_size)

            # calculate hyper-volume in case of multi-objective optimization
            if gen == ngen:
                hyp_volume = hypervolume(valid, [1.0, 1.0, 1.0, 1.0])
        else:
            if gen == 1:
                print( "MU and length of valid", MU, len(valid))
                offspring = toolbox.select(valid, MU)
            else:
                offspring = toolbox.select(valid + old_population, MU)

            if gen == ngen:
                hyp_volume = hypervolume(valid, [1.0, 1.0, 1.0, 1.0])

        end = time.time()
        selection_time = end - start
        lexicase_cases = [ind.n_cases for ind in offspring]
        avg_n_cases = sum(lexicase_cases) / len(lexicase_cases)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth,
                           codon_consumption, invalidate_max_depth,
                           genome_representation, max_genome_length)

        # Evaluate the individuals with an invalid fitness
        evaluated_inds = 0
        for ind in offspring:
            # Now, we evaluate the individual
            # Note that the individual can also be invalid for other reasons
            # so we always need to consider this in the fitness function
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
                # print(ind.fitness.values)
                evaluated_inds += 1

        # Update population for next generation
        if algo != "weighted":
            old_population = population.copy()
            old_population = [ind for ind in old_population if not ind.invalid]
            old_population = [ind for ind in old_population if not math.isnan(ind.fitness.values[0])]

        population[:] = offspring
        # Include in the population the elitist individuals
        for i in range(elite_size):
            population.append(halloffame.items[i])

        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]

        if len(valid0) != len(valid):
            warnings.warn(
                "Warning: There are valid individuals with fitness = NaN in the population. We will avoid in the statistics and selection process.")
        invalid = len(population) - len(
            valid0)  # We use the original number of invalids in this case, because we just want to count the completely mapped individuals

        list_structures = []

        # for ind in offspring:
        for idx, ind in enumerate(valid):
            # if ind.invalid == True:
            #    invalid += 1
            # else:
            list_structures.append(str(ind.structure))

        unique_structures = np.unique(list_structures, return_counts=False)

        structural_diversity = len(unique_structures) / len(population)

        length = [len(ind.genome) for ind in valid]
        avg_length = sum(length) / len(length)

        nodes = [ind.nodes for ind in valid]
        avg_nodes = sum(nodes) / len(nodes)

        depth = [ind.depth for ind in valid]
        avg_depth = sum(depth) / len(depth)

        used_codons = [ind.used_codons for ind in valid]
        avg_used_codons = sum(used_codons) / len(used_codons)

        end_gen = time.time()
        generation_time = end_gen - start_gen

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if not verbose:
                print("gen =", gen, " Fitness =", halloffame.items[0].fitness.values, "best_ind depth = ",
                      best_ind_depth, ", Invalids =", invalid)
            if points_test:
                if gen < ngen:
                    fitness_test = np.NaN
                else:
                    fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, invalid=invalid, train_fitness=halloffame.items[0].fitness.values,
                       best_ind_length=best_ind_length, avg_length=avg_length,
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       structural_diversity=structural_diversity,
                       evaluated_inds=evaluated_inds,
                       avg_n_cases=avg_n_cases)

        # Check whether the specified path exists or not
        isExist = os.path.exists(final_gen_file_location)
        if not isExist:
            os.makedirs(final_gen_file_location)

        if save_final_gen:
            if gen == ngen:
                o1 = []
                o2 = []
                o3 = []
                o4 = []
                phenotype = []
                for indi in valid:
                    # value = toolbox.evaluate(indi, points_train)
                    # o1.append(value[0])
                    # o2.append(value[0])
                    # o3.append(value[0])
                    # o4.append(value[0])
                    o1.append(indi.o_1)
                    o2.append(indi.o_2)
                    o3.append(indi.o_3)
                    o4.append(indi.o_4)
                    phenotype.append(indi.phenotype)

                data_dict = {"Phenotype": phenotype, "o_1": o1, "o_2": o2, "o_3": o3, "o_4": o4}
                df = pd.DataFrame(data_dict)
                df.to_csv(final_gen_file_location + "final_gen.csv")



        # save individuals after every 20 generation
        if save_every_20_gen:
            if gen%20 == 0:
                o1 = []
                o2 = []
                o3 = []
                o4 = []
                phenotype = []
                for indi in valid:
                    o1.append(indi.o_1)
                    o2.append(indi.o_2)
                    o3.append(indi.o_3)
                    o4.append(indi.o_4)
                    phenotype.append(indi.phenotype)

                data_dict = {"Phenotype": phenotype, "o_1": o1, "o_2": o2, "o_3": o3, "o_4": o4 }
                df = pd.DataFrame(data_dict)
                df.to_csv(final_gen_file_location + str(gen) + "_gen.csv")


        if verbose:
            print(logbook.stream)

    if algo != "weighted":
        return population, logbook, hyp_volume
    else:
        return population, logbook, None

