# -*- coding: utf-8 -*-
import torch

import grape
import optimization_algorithm

import pandas as pd
import numpy as np
from deap import creator, base, tools
import random
import csv
import pandas as pd
import torch.nn as nn
import gower
import textwrap
from pickle import dump , load

# from functions import add, sub, mul, pdiv
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
# import sys

import os
import copy

from sklearn.neighbors import NearestNeighbors
from math import factorial
# Suppressing Warnings:
import warnings
import multiprocessing
from functools import partial


warnings.filterwarnings("ignore")

# create   a function to import input  data in numpy format : x_train , y_train
# if y = 0 , counter-factual will try to create y = 1
class TaiwaneseCreditClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(23, 64)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 128)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(128, 32)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(32, 16)
        self.act4 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        self.act_output = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.act1(self.hidden1(x)))
        x = self.dropout(self.act2(self.hidden2(x)))
        x = self.dropout(self.act3(self.hidden3(x)))
        x = self.dropout(self.act4(self.hidden4(x)))
        # x = self.output(x)
        x = self.act_output(self.output(x))
        return x


def scale_input_data(scaler, df, columns_to_standardize):
    numpy_array = torch.tensor(df.iloc[:, 0:23].values, dtype=torch.float32)
    numpy_array = numpy_array.numpy()
    numpy_array[:, columns_to_standardize] = scaler.transform(numpy_array[:, columns_to_standardize])
    return numpy_array


def load_scaler(scaler_loc):
    return load(open(scaler_loc, 'rb'))


def process_data(df, model, scaler, columns_to_standardize):
    scaled = scale_input_data(scaler, df, columns_to_standardize)
    with torch.no_grad():
        y_pred = model(torch.tensor(scaled, dtype=torch.float32))
    probability = [float(x) for xs in y_pred.tolist() for x in xs]
    # print(probability)
    output = y_pred.round().tolist()
    predictions = [int(x) for xs in output for x in xs]
    df['predictions'] = predictions
    df['probability'] = probability
    df_1 = df[df['predictions'] == 1]
    df_0 = df[df['predictions'] == 0]
    df_0 = df_0.sort_values(by='probability', ascending=True)
    df_1 = df_1.sort_values(by='probability', ascending=False)
    # print(df_0.head())
    print("shape of dataframe with prediction as 1:", df_1.shape)
    print("shape of dataframe with prediction as 0:", df_0.shape)
    input_0 = df_0.iloc[:, 0:23].values
    input_1 = df_1.iloc[:, 0:23].values
    return input_0, input_1

def check_bounds(np_array, bounds):
    """
    Check if elements in the numpy array are within the given bounds.

    Parameters:
    np_array (np.ndarray): Input numpy array.
    bounds (list of tuple): List of (min, max) bounds for each index.

    Returns:
    bool: True if all elements are within the bounds, False otherwise.
    """
    # Ensure the input is a NumPy array
    np_array = np.asarray(np_array)

    # Check if the length of bounds matches the length of the array
    if len(np_array) != len(bounds):
        raise ValueError("Length of bounds must match the length of the numpy array")

    # Iterate over each element and its corresponding bounds
    for idx, (value, (min_bound, max_bound)) in enumerate(zip(np_array, bounds)):
        if not (min_bound <= value <= max_bound):
            return False

    return True


def filter_dataframe(df, bounds, drop_columns=[]):
    """
    Filter rows of a DataFrame based on bounds for each column and drop specified columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    bounds (list of tuple): List of (min, max) bounds for each column.
    drop_columns (list of str): List of column names to be dropped.

    Returns:
    pd.DataFrame: Filtered DataFrame with rows within bounds and specified columns dropped.
    """
    # Drop specified columns
    df = df.drop(columns=drop_columns)

    # Validate bounds length
    if len(bounds) != len(df.columns):
        raise ValueError("Length of bounds must match the number of remaining DataFrame columns")

    # Apply the check_bounds function to each row and filter the DataFrame
    filtered_df = df[df.apply(lambda row: check_bounds(row.values, bounds), axis=1)]

    return filtered_df


def convert_int_to_float(df):
    # Select columns with integer data types
    int_columns = df.select_dtypes(include=['int'])

    # Convert all integer columns to float
    df[int_columns.columns] = int_columns.astype(float)

    # print("All integer columns have been converted to float.")
    return df


def create_input_data(data_file_loc, grammar_loc, model, bounds, scaler, columns_to_standardize ):
    df = pd.read_csv(data_file_loc)
    # print(df.columns)
    df = filter_dataframe(df, bounds, drop_columns=["Y"])
    input_0, input_1 = process_data(df, model,scaler, columns_to_standardize)
    BNF_GRAMMAR = grape.Grammar(grammar_loc)
    return input_0, input_1, BNF_GRAMMAR


def load_model():
    model = torch.load('model_training/Taiwanese_credit_model')
    model.eval()
    return model


def eval_model(model, input, scaler, columns_to_standardize):
    model.eval()
    # print(input)
    input = input.reshape(-1, 23)
    input = torch.tensor(input, dtype=torch.float32)
    input = input.numpy()
    input[:, columns_to_standardize] = scaler.transform(input[:, columns_to_standardize])
    input = torch.from_numpy(input).type(torch.float)
    with torch.no_grad():
        prob = model(input)
    return 1 - prob.tolist()[0][0]


def manhattan(a, b):
    # print('old x {}, changed x {}'.format(a, b))
    return sum(abs(a-b))


def convert_int_to_float_np(array):
    return array.astype(float)


def calculate_gower_distance(a, b, min, max):
    # print("The values inside functions are: ", a, b, min, max)
    # print(np.array([a, b, min, max]))
    input_array = convert_int_to_float_np(np.array([a, b, min, max]))
    # print(input_array)
    # exit()
    # mat = gower.gower_matrix(np.array([a, b, min, max ]))
    mat = gower.gower_matrix(input_array)
    return mat[0][1]


def changed_count_distance(input_0, input_1):
    changed_count = np.sum(input_0 != input_1)
    return changed_count/len(input_0)


def out_of_distribution_distance(data, query_point, k=5):
    # Convert data to numpy arrays if not already
    data = np.asarray(data)
    query_point = np.asarray(query_point)
    # Initialize NearestNeighbors object
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    # Find indices of k-nearest neighbors
    distances, indices = nbrs.kneighbors([query_point])
    # Calculate average Euclidean distance from all neighbors
    avg_distance = np.mean(distances)
    return avg_distance


def out_of_distribution_distance_gower(data, query_point, k=7):
    query_point = convert_int_to_float_np(query_point)
    data = convert_int_to_float(data)
    df_query = pd.DataFrame(data=[query_point])
    distances = gower.gower_topn(df_query, data, n=k)
    return distances['values'].mean()


def fitness_eval_MOC( individual, points, model, multiobj, bounds, index,  scaler, columns_to_standardize):
    global x
    whole_dataset = copy.deepcopy(points)
    df = pd.DataFrame(data= whole_dataset)
    min = df.min().to_list()
    max= df.max().to_list()
    x = copy.deepcopy(points[index])
    x_original = copy.deepcopy(x)
    if individual.invalid == True:
        if multiobj:
            return np.NaN, np.NaN, np.NaN, np.NaN
        else:
            return np.NaN

    else:
        try:
            exec(individual.phenotype, globals())

            # print('updated x = {} and original x = {}'.format(x, x_original))# update features

            # check bounds
            if not check_bounds(x, bounds):
                # print("Individual out of bound")
                if multiobj:
                    # return np.NaN, np.NaN, np.NaN, np.NaN
                    return 1000, 1000, 1000, 1000
                else:
                    # return np.NaN
                    return 1000
            else:

                # maximize o_1 for class 0
                # model = load_model()
                # o_1 = eval_model(model, torch.tensor(x, dtype=torch.float32))
                o_1 = eval_model(model, x,  scaler, columns_to_standardize)
                individual.o_1 = o_1
                # print(o_1)

                # for calculating o2 we need the parent within this function
                # o_2 = manhattan(x_original, x)
                o_2 = calculate_gower_distance(x_original, x, min, max)
                individual.o_2 = o_2

                # for o3 we will calculate how many variable changed
                o_3 = changed_count_distance(x_original, x)
                individual.o_3 = o_3
                # for o4 we need to define a function that penalizes for out_of_distribution shift in variable
                # o_4 = out_of_distribution_distance(whole_dataset, x)

                o_4 = out_of_distribution_distance_gower(df, x)
                individual.o_4 = o_4

        except (FloatingPointError, ZeroDivisionError, OverflowError,MemoryError, ValueError, TypeError) as e:
            print("got exception: ", e)
            if multiobj:
                return np.NaN, np.NaN, np.NaN, np.NaN
            else:
                return np.NaN
        if multiobj:
            return o_1, o_2, o_3, o_4
        else:
            w1, w2, w3, w4 = 0.79, 0.005, 0.005, 0.2
            return w1*o_1 + w2*o_2 + w3*o_3 + w4*o_4


def generate_counterfactuals(input, algo, fitness_function, multiobj, bounds, ref_points,\
                             N_RUNS, run, pop_size, BNF_GRAMMAR,  MIN_INIT_TREE_DEPTH,  MAX_INIT_TREE_DEPTH, CODON_SIZE, CODON_CONSUMPTION,\
                             GENOME_REPRESENTATION, HALLOFFAME_SIZE, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS, ELITE_SIZE, MAX_TREE_DEPTH,\
                             MAX_GENOME_LENGTH, REPORT_ITEMS, problem,  index, save_file_location):
    toolbox = base.Toolbox()
    model = load_model()
    columns_to_standardize = list(range(23))

    scaler = load_scaler(scaler_loc='model_training/StandardScaler.pkl')

    if algo == 'weighted':
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        toolbox.register("select", tools.selTournament, tournsize=7)
    elif algo == "NSGAII":
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        toolbox.register("select", tools.selNSGA2)
    else:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)
    toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
    # Single-point crossover:
    toolbox.register("mate", grape.crossover_onepoint)
    # Flip-int mutation:
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)
    toolbox.register("evaluate", fitness_function, model=model, multiobj=multiobj, bounds=bounds, index=index, scaler=scaler, columns_to_standardize=columns_to_standardize)

    print("the (1 - probability) of the current input is :", eval_model(model, input[index], scaler, columns_to_standardize))

    hyp_volume_list = []

    for i in range(N_RUNS):
        print(2 * "\n")
        print("Run:", i + run)

        RANDOM_SEED = i + run

        random.seed(RANDOM_SEED)


        population = toolbox.populationCreator(pop_size=pop_size,
                                               bnf_grammar=BNF_GRAMMAR,
                                               min_init_depth=MIN_INIT_TREE_DEPTH,
                                               max_init_depth=MAX_INIT_TREE_DEPTH,
                                               codon_size=CODON_SIZE,
                                               codon_consumption=CODON_CONSUMPTION,
                                               genome_representation=GENOME_REPRESENTATION
                                               )

        hof = tools.HallOfFame(HALLOFFAME_SIZE)

        if algo=="NSGAIII":
            MU= pop_size
        else:
            MU= None

        # print("The value of MU: ", MU)
        population, logbook, hyp_volume = optimization_algorithm.ge_eaSimpleWithElitism(population, toolbox,
                                                                                        cxpb=P_CROSSOVER,
                                                                                        mutpb=P_MUTATION,
                                                                                        ngen=MAX_GENERATIONS,
                                                                                        elite_size=ELITE_SIZE,
                                                                                        bnf_grammar=BNF_GRAMMAR,
                                                                                        codon_size=CODON_SIZE,
                                                                                        max_tree_depth=MAX_TREE_DEPTH,
                                                                                        max_genome_length=MAX_GENOME_LENGTH,
                                                                                        points_train=input,
                                                                                        codon_consumption=CODON_CONSUMPTION,
                                                                                        report_items=REPORT_ITEMS,
                                                                                        genome_representation=GENOME_REPRESENTATION,
                                                                                        invalidate_max_depth=False,
                                                                                        problem=problem,
                                                                                        halloffame=hof,
                                                                                        final_gen_file_location=save_file_location,
                                                                                        save_final_gen=True,
                                                                                        save_every_20_gen=False,
                                                                                        verbose=False, algo=algo, MU=MU)

        hyp_volume_list.append(hyp_volume)

        print("hyper_volume: ", hyp_volume_list)

        best = hof.items[0].phenotype
        print("Best individual: \n", "\n".join(textwrap.wrap(best, 80)))
        print("\nTraining Fitness: ", hof.items[0].fitness.values[0])

        print("Depth: ", hof.items[0].depth)
        print("Length of the genome: ", len(hof.items[0].genome))
        print(f'Used portion of the genome: {hof.items[0].used_codons / len(hof.items[0].genome):.2f}')

        train_fitness = logbook.select("train_fitness")

        best_ind_length = logbook.select("best_ind_length")
        avg_length = logbook.select("avg_length")

        gen, invalid = logbook.select("gen", "invalid")
        avg_used_codons = logbook.select("avg_used_codons")
        best_ind_used_codons = logbook.select("best_ind_used_codons")

        best_ind_nodes = logbook.select("best_ind_nodes")
        avg_nodes = logbook.select("avg_nodes")

        best_ind_depth = logbook.select("best_ind_depth")
        avg_depth = logbook.select("avg_depth")

        structural_diversity = logbook.select("structural_diversity")
        evaluated_inds = logbook.select("evaluated_inds")

        best_phenotype = [float('nan')] * MAX_GENERATIONS
        best_phenotype.append(best)

        r = RANDOM_SEED

        header = REPORT_ITEMS

        address = save_file_location + str("log_records") + "/"

        # Check whether the specified path exists or not
        isExist = os.path.exists(address)
        if not isExist:
            os.makedirs(address)

        with open(address + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(header)
            for value in range(len(avg_nodes)):
                writer.writerow([gen[value], invalid[value],
                                 train_fitness[value],
                                 best_ind_length[value],
                                 avg_length[value],
                                 best_ind_nodes[value],
                                 avg_nodes[value],
                                 best_ind_depth[value],
                                 avg_depth[value],
                                 avg_used_codons[value],
                                 best_ind_used_codons[value],
                                 structural_diversity[value],
                                 best_phenotype[value]])

    data_dict = {"hyper_volume": hyp_volume_list}
    df = pd.DataFrame(data_dict)
    df.to_csv(save_file_location + "hyper_volume.csv")
    np.save(save_file_location + "input_data.npy", input[index])


def generate_counterfactuals_wrapper(index, save_file_location, input_0, algo, fitness_eval_MOC,
                                     multiobj, bounds, ref_points, N_RUNS, run, MU, BNF_GRAMMAR,
                                     MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH, CODON_SIZE, CODON_CONSUMPTION,
                                     GENOME_REPRESENTATION, HALLOFFAME_SIZE, P_CROSSOVER, P_MUTATION, MAX_GENERATIONS,
                                     ELITE_SIZE, MAX_TREE_DEPTH, MAX_GENOME_LENGTH, REPORT_ITEMS, problem):
    save_file = save_file_location + str(index) + "/"
    generate_counterfactuals(input=input_0,
                             algo=algo,
                             fitness_function=fitness_eval_MOC,
                             multiobj=multiobj,
                             bounds=bounds,
                             ref_points=ref_points,
                             N_RUNS=N_RUNS,
                             run=run,
                             pop_size=MU,
                             BNF_GRAMMAR=BNF_GRAMMAR,
                             MIN_INIT_TREE_DEPTH=MIN_INIT_TREE_DEPTH,
                             MAX_INIT_TREE_DEPTH=MAX_INIT_TREE_DEPTH,
                             CODON_SIZE=CODON_SIZE,
                             CODON_CONSUMPTION=CODON_CONSUMPTION,
                             GENOME_REPRESENTATION=GENOME_REPRESENTATION,
                             HALLOFFAME_SIZE=HALLOFFAME_SIZE,
                             P_CROSSOVER=P_CROSSOVER,
                             P_MUTATION=P_MUTATION,
                             MAX_GENERATIONS=MAX_GENERATIONS,
                             ELITE_SIZE=ELITE_SIZE,
                             MAX_TREE_DEPTH=MAX_TREE_DEPTH,
                             MAX_GENOME_LENGTH=MAX_GENOME_LENGTH,
                             REPORT_ITEMS=REPORT_ITEMS,
                             problem=problem,
                             index=index,
                             save_file_location=save_file)



if __name__ == "__main__":

    problem = 'Taiwanese_credit_default'
    scenario = 0
    run = 1
    N_RUNS = 30

    multiobj = True
    algo = "NSGAII"
    save_file_location = "output/" + algo + "_multi/"

    # NSGA-III specific Algorithm parameters

    NOBJ = 4
    P = 12
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
    MU = int(H + (4 - H % 4))
    # print(MU)
    # Create uniform reference point
    ref_points = tools.uniform_reference_points(NOBJ, P)
    # print(ref_points)

    # Evolutionary parameters

    POPULATION_SIZE = 456
    MAX_INIT_TREE_DEPTH = 20
    MIN_INIT_TREE_DEPTH = 4

    MAX_GENERATIONS = 100  # 0
    P_CROSSOVER = 0.8
    P_MUTATION = 0.01
    ELITE_SIZE = 1
    HALLOFFAME_SIZE = 1

    CODON_CONSUMPTION = 'lazy'
    GENOME_REPRESENTATION = 'list'
    MAX_GENOME_LENGTH = None  # 'auto'

    MAX_TREE_DEPTH = 17  # equivalent to 17 in GP with this grammar
    MAX_WRAPS = 0
    CODON_SIZE = 255

    REPORT_ITEMS = ['gen', 'invalid', 'train_fitness',
                    'best_ind_length', 'avg_length',
                    'best_ind_nodes', 'avg_nodes',
                    'best_ind_depth', 'avg_depth',
                    'avg_used_codons', 'best_ind_used_codons',
                    'structural_diversity',
                    'best_ind_phenotype']

    bounds = [(10000, 1000000),
              (0, 1),
              (0, 6),
              (0, 3),
              (20, 80),
              (-2, 8),
              (-2, 8),
              (-2, 8),
              (-2, 8),
              (-2, 8),
              (-2, 8),
              (-154973, 964511),
              (-67526, 983931),
              (-61506, 855086),
              (-81334, 891586),
              (53007, 927171),
              (-339603, 961664),
              (0, 873552),
              (0, 1215471),
              (0, 846040),
              (0, 621000),
              (0, 417990),
              (0, 403500)]

    columns_to_standardize = list(range(23))

    scaler = load_scaler(scaler_loc='model_training/StandardScaler.pkl')


    model = load_model()
    input_0, input_1, BNF_GRAMMAR = create_input_data("test_balanced.csv",
                                                      "./../grammars/taiwanese_dataset.bnf",
                                                      model, bounds, scaler, columns_to_standardize )

    num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_processes) as pool:
        pool.map(partial(generate_counterfactuals_wrapper,
                         save_file_location=save_file_location,
                         input_0=input_0,
                         algo=algo,
                         fitness_eval_MOC=fitness_eval_MOC,
                         multiobj=multiobj,
                         bounds=bounds,
                         ref_points=ref_points,
                         N_RUNS=N_RUNS,
                         run=run,
                         MU=MU,
                         BNF_GRAMMAR=BNF_GRAMMAR,
                         MIN_INIT_TREE_DEPTH=MIN_INIT_TREE_DEPTH,
                         MAX_INIT_TREE_DEPTH=MAX_INIT_TREE_DEPTH,
                         CODON_SIZE=CODON_SIZE,
                         CODON_CONSUMPTION=CODON_CONSUMPTION,
                         GENOME_REPRESENTATION=GENOME_REPRESENTATION,
                         HALLOFFAME_SIZE=HALLOFFAME_SIZE,
                         P_CROSSOVER=P_CROSSOVER,
                         P_MUTATION=P_MUTATION,
                         MAX_GENERATIONS=MAX_GENERATIONS,
                         ELITE_SIZE=ELITE_SIZE,
                         MAX_TREE_DEPTH=MAX_TREE_DEPTH,
                         MAX_GENOME_LENGTH=MAX_GENOME_LENGTH,
                         REPORT_ITEMS=REPORT_ITEMS,
                         problem=problem),
                 range(0, 30))




