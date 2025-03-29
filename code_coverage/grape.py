# -*- coding: utf-8 -*-


import re
import math
from operator import attrgetter
import numpy as np
import random
import copy

#from scipy.stats import median_abs_deviation
#from statsmodels import robust   

def median_abs_deviation(arr, axis=0):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Calculate the median along axis 0
    median = np.median(arr, axis=0)

    # Calculate the absolute deviations from the median along axis 0
    abs_deviations = np.abs(arr - median)

    # Calculate the median of the absolute deviations along axis 0
    mad = np.median(abs_deviations, axis=0)

    return mad

class Individual(object):
    """
    A GE individual.
    """

    def __init__(self, genome, grammar, max_depth, codon_consumption):
        """
        """
        
        self.genome = genome
        self.length = len(genome)
        if codon_consumption == 'lazy':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_lazy(genome, grammar, max_depth)
        elif codon_consumption == 'eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_eager(genome, grammar, max_depth)
        elif codon_consumption == 'leap':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.tile_size, \
            self.effective_positions = mapper_leap(genome, grammar, max_depth)
        elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.tile_size, \
            self.effective_positions = mapper_leap2(genome, grammar, max_depth)
        elif codon_consumption == 'multiGE':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_multi(genome, grammar, max_depth)
        elif codon_consumption == 'multichromosomalGE':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_multichromosomal(genome, grammar, max_depth)            
        elif codon_consumption == 'parameterised':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_parameterised(genome, grammar, max_depth)    
        elif codon_consumption == 'cosmo_eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.cosmo = mapper_cosmo(genome, grammar, max_depth)
        elif codon_consumption == 'cosmo_eager_depth':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.cosmo = mapper_cosmo_ext(genome, grammar, max_depth)       
        elif codon_consumption == 'cosmo_total':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.cosmo = mapper_cosmo_total(genome, grammar, max_depth)     
        else:
            raise ValueError("Unknown mapper")
            
        self.fitness_each_sample = []
        self.n_cases = 0

class Grammar(object):
    """
    Attributes:
    - non_terminals: list with each non-terminal (NT);
    - start_rule: first non-terminal;
    - production_rules: list with each production rule (PR), which contains in each position:
        - the PR itself as a string
        - 'non-terminal' or 'terminal'
        - the arity (number of NTs in the PR)
        - production choice label
        - True, if it is recursive, and False, otherwise
        - the minimum depth to terminate the mapping of all NTs of this PR
        - the minimum number of codons to terminate the mapping of all NTs of this PR
    - n_rules: df
    - max_codons_each_PR
    - min_codons_each_PR
    - max_depth_each_PR
    - initial_next_NT
    - initial_list_depth
    - initial_list_codons
    
    """
    def __init__(self, file_address):
        #Reading the file
        with open(file_address, "r") as text_file:
            bnf_grammar = text_file.read()
        #Getting rid of all the duplicate spaces
        bnf_grammar = re.sub(r"\s+", " ", bnf_grammar)

        #self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>\s*::=",bnf_grammar)]
        self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>\s*::=",bnf_grammar)]
        self.start_rule = self.non_terminals[0]
        for i in range(len(self.non_terminals)):
            bnf_grammar = bnf_grammar.replace(self.non_terminals[i] + " ::=", "  ::=")
        rules = bnf_grammar.split("::=")
        del rules[0]
        rules = [item.replace('\n',"") for item in rules]
        rules = [item.replace('\t',"") for item in rules]
        
        #list of lists (set of production rules for each non-terminal)
        self.production_rules = [i.split('|') for i in rules]
        for i in range(len(self.production_rules)):
            #Getting rid of all leading and trailing whitespaces
            self.production_rules[i] = [item.strip() for item in self.production_rules[i]]
            for j in range(len(self.production_rules[i])):
                #Include in the list the PR itself, NT or T, arity and the production choice label
                #if re.findall(r"\<(\w+)\>",self.production_rules[i][j]):
                if re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]):                    
                    #arity = len(re.findall(r"\<(\w+)\>",self.production_rules[i][j]))
                    arity = len(re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]))
                    self.production_rules[i][j] = [self.production_rules[i][j] , "non-terminal", arity, j]
                else:
                    self.production_rules[i][j] = [self.production_rules[i][j] , "terminal", 0, j] #arity 0
        #number of production rules for each non-terminal
        self.n_rules = [len(list_) for list_ in self.production_rules]
        


#        check_recursiveness = []
#        for PR in reversed(self.production_rules):
#            idx = self.production_rules.index(PR)
#            for j in range(len(PR)):
#                items = re.findall(r"\<([\(\)\w,-.]+)\>", PR[j][0])
#                for NT in items:
#                    if (NT not in check_recursiveness):
#                        for k in range(0, idx): #len(self.production_rules)
#                            for l in range(len(self.production_rules[k])):
#                                if NT in self.production_rules[k][l][0]:
#                                    check_recursiveness.append(self.non_terminals[k])


        
#        check_recursiveness = []
##        for i in range(len(self.production_rules)):
 #           for j in range(len(self.production_rules[i])):
 #               items = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])
                #n_non_terminals = len(items)
 #               for NT in items:
 #                   if (NT not in check_recursiveness) and (i < len(self.production_rules) - 1):
 #                       for k in range(i + 1, len(self.production_rules)):
 #                           for l in range(0, len(self.production_rules[k])):
 #                               if NT in self.production_rules[k][l][0]:
 #                                   check_recursiveness.append(self.non_terminals[i])
                
            
        
#begin        
        #Building list of non-terminals with recursive production-rules
#        check_recursiveness = []
#        try_recursiveness = self.non_terminals.copy()
#        recursive_indexes = []
#        for i in range(len(self.production_rules)):
#            check = 0
#            for j in range(len(self.production_rules[i])):
#                if self.production_rules[i][j][1] == 'non-terminal':
 #                   check += 1
 #           if check == 0: #if the PR has only terminals, it is not possible to be recursive
 #               if self.non_terminals[i] in try_recursiveness:
 #                   try_recursiveness.remove(self.non_terminals[i])
 #           else: #the PR has at least one recursive choice and it is therefore recursive
 #               recursive_indexes.append(i)
        
#        for item in reversed(try_recursiveness):
#            idx = self.non_terminals.index(item)
#            for i in range(len(self.production_rules[idx])):
#                if item in self.production_rules[idx][i][0]:
#                    if item not in check_recursiveness:
#                        check_recursiveness.append(item)
#                for recursive_item in check_recursiveness:
#                    if recursive_item in self.production_rules[idx][i][0]:
#                        if recursive_item not in check_recursiveness:
#                            check_recursiveness.append(recursive_item)
                
#        check_size = len(check_recursiveness) - 1
#        while check_size != len(check_recursiveness):
#            check_size = len(check_recursiveness)
 #           for item in check_recursiveness:
  
 #               for i in range(len(self.production_rules)):
 #                   for j in range(len(self.production_rules[i])):
 #                       if item in self.production_rules[i][j][0]:
 #                           if self.non_terminals[i] not in check_recursiveness:
 #                               check_recursiveness.append(self.non_terminals[i])
#end

#                        for recursive_item in check_recursiveness:
#                            if recursive_item in self.production_rules[idx][i][0]:
#                                if recursive_item not in check_recursiveness:
#                                    check_recursiveness.append(recursive_item)
                
            
        
        
#        check_recursiveness = []
#        check_size = len(try_recursiveness)
#        continue_ = True
#        while continue_:
#            for k in range(len(try_recursiveness)):
#                for i in range(len(self.production_rules)):
#                    for j in range(len(self.production_rules[i])):
#                        if i >= k:
#                            if try_recursiveness[k] in self.production_rules[i][j][0]:
#                                if self.non_terminals[i] not in check_recursiveness:
#                                    check_recursiveness.append(self.non_terminals[i])
#                                    if self.non_terminals[i] == '<nonboolean_feature>':
#                                        pass
#            if len(check_recursiveness) != check_size:
#                check_size = len(check_recursiveness)
#            else:
#                continue_ = False
                
        #Building list of non-terminals with recursive production-rules
#        try_recursiveness = self.non_terminals
#        check_recursiveness = []
#        check_size = len(try_recursiveness)
#        continue_ = True
#        while continue_:
#            for k in range(len(try_recursiveness)):
#                for i in range(len(self.production_rules)):
#                    for j in range(len(self.production_rules[i])):
#                        if i >= k:
#                            if try_recursiveness[k] in self.production_rules[i][j][0]:
#                                if self.non_terminals[i] not in check_recursiveness:
#                                    check_recursiveness.append(self.non_terminals[i])
#                                    if self.non_terminals[i] == '<nonboolean_feature>':
#                                        pass
#            if len(check_recursiveness) != check_size:
#                check_size = len(check_recursiveness)
#            else:
#                continue_ = False


            
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[i][j][0])
                NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
                unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
                recursive = False
      #          while unique_NTs.size and not recursive:
                for NT_to_check in unique_NTs:
                    stack = [self.non_terminals[i]]  
                    if NT_to_check in stack:
                        recursive = True
                        break
                    else:
                        #check_recursiveness.append(NT_to_check)
                        stack.append(NT_to_check)
                        recursive = check_recursiveness(self, NT_to_check, stack)
                        if recursive:
                            break
                        stack.pop()
                self.production_rules[i][j].append(recursive)
                

    
        

#finished

#        check_recursiveness = [self.start_rule]
#        check_size = len(check_recursiveness)
#        continue_ = True
#        while continue_:
#            for i in range(len(self.production_rules)):
#                for j in range(len(self.production_rules[i])):
#                    for k in range(len(check_recursiveness)):
#                        if check_recursiveness[k] in self.production_rules[i][j][0]:
#                            if self.non_terminals[i] not in check_recursiveness:
#                                check_recursiveness.append(self.non_terminals[i])
#            if len(check_recursiveness) != check_size:
#                check_size = len(check_recursiveness)
#            else:
#                continue_ = False

        
        #Including information of recursiveness in each production-rule
        #True if the respective non-terminal has recursive production-rules. False, otherwise
#        for i in range(len(self.production_rules)):
#            for j in range(len(self.production_rules[i])):
#                if self.production_rules[i][j][1] == 'terminal': #a terminal is never recursive
#                    recursive = False
#                    self.production_rules[i][j].append(recursive)
#                else: #a non-terminal can be recursive
#                    for k in range(len(check_recursiveness)):
#                        #Check if a recursive NT is in the current list of PR
                        #TODO I just changed from self.non_terminals[k] to check_recursiveness[k]
#                        if check_recursiveness[k] in self.production_rules[i][j][0]:
#                            recursive = True
#                            break #since we already found a recursive NT in this PR, we can stop
#                        else:
#                            recursive = False #if there is no recursive NT in this PR
#                    self.production_rules[i][j].append(recursive)
        
        #minimum depth from each non-terminal to terminate the mapping of all symbols
        NT_depth_to_terminate = [None]*len(self.non_terminals)
        #minimum depth from each production rule to terminate the mapping of all symbols
        part_PR_depth_to_terminate = list() #min depth for each non-terminal or terminal to terminate
        isolated_non_terminal = list() #None, if the respective position has a terminal
        #Separating the non-terminals within the same production rule
        for i in range(len(self.production_rules)):
            part_PR_depth_to_terminate.append( list() )
            isolated_non_terminal.append( list() )
            for j in range(len(self.production_rules[i])):
                part_PR_depth_to_terminate[i].append( list() )
                isolated_non_terminal[i].append( list() )
                if self.production_rules[i][j][1] == 'terminal':
                    isolated_non_terminal[i][j].append(None)
                    part_PR_depth_to_terminate[i][j] = 1
                    if not NT_depth_to_terminate[i]:
                        NT_depth_to_terminate[i] = 1
                else:
                    for k in range(self.production_rules[i][j][2]): #arity
                        part_PR_depth_to_terminate[i][j].append( list() )
                        #term = re.findall(r"\<(\w+)\>",self.production_rules[i][j][0])[k]
                        term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
                        isolated_non_terminal[i][j].append('<' + term + '>')
        continue_ = True
        while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
            if None not in NT_depth_to_terminate:
                continue_ = False 
            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(isolated_non_terminal[j][k])):
                            if self.non_terminals[i] == isolated_non_terminal[j][k][l]:
                                if NT_depth_to_terminate[i]:
                                    if not part_PR_depth_to_terminate[j][k][l]:
                                        part_PR_depth_to_terminate[j][k][l] = NT_depth_to_terminate[i] + 1
                                        if [] not in part_PR_depth_to_terminate[j][k]:
                                            if not NT_depth_to_terminate[j]:
                                                NT_depth_to_terminate[j] = part_PR_depth_to_terminate[j][k][l]
        PR_depth_to_terminate = []
        for i in range(len(part_PR_depth_to_terminate)):
            for j in range(len(part_PR_depth_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
                if type(part_PR_depth_to_terminate[i][j]) == int:
                    depth_ = part_PR_depth_to_terminate[i][j]
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                else:
                    depth_ = max(part_PR_depth_to_terminate[i][j])
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                    
        #minimum number of codons from each non-terminal to terminate the mapping of all symbols
        NT_codons_to_terminate = [None]*len(self.non_terminals)
        #minimum number of codons from each production rule to terminate the mapping of all symbols
        part_PR_codons_to_terminate = list() #min number of codons for each non-terminal or terminal to terminate
        codons_non_terminal = list() #None, if the respective position has a terminal
        #Separating the non-terminals within the same production rule
 #       for i in range(len(self.production_rules)):
 #           part_PR_codons_to_terminate.append( list() )
 #           codons_non_terminal.append( list() )
 #           for j in range(len(self.production_rules[i])):
 #               part_PR_codons_to_terminate[i].append( list() )
 #               codons_non_terminal[i].append( list() )
 #               if self.production_rules[i][j][1] == 'terminal':
 #                   codons_non_terminal[i][j].append(None)
 #                   part_PR_codons_to_terminate[i][j] = 1
 #                   if not NT_codons_to_terminate[i]:
 #                       NT_codons_to_terminate[i] = 1
 #               else:
 #                   for k in range(self.production_rules[i][j][2]): #arity
 #                       part_PR_codons_to_terminate[i][j].append( list() )
 #                       term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
 #                       codons_non_terminal[i][j].append('<' + term + '>')
 #       continue_ = True
 #       while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
 #           if None not in NT_codons_to_terminate:
 #               continue_ = False 
 #           for i in range(len(self.non_terminals)):
 #               for j in range(len(self.production_rules)):
 #                   for k in range(len(self.production_rules[j])):
 #                       for l in range(len(codons_non_terminal[j][k])):
 #                           if self.non_terminals[i] == codons_non_terminal[j][k][l]:
 #                               if NT_codons_to_terminate[i]:
 #                                   if not part_PR_codons_to_terminate[j][k][l]:
 #                                       part_PR_codons_to_terminate[j][k][l] = NT_codons_to_terminate[i] + 1
 #                                       if [] not in part_PR_codons_to_terminate[j][k]:
 #                                           if not NT_codons_to_terminate[j]:
 #                                               NT_codons_to_terminate[j] = part_PR_codons_to_terminate[j][k][l]
 #       PR_codons_to_terminate = []
 #       for i in range(len(part_PR_codons_to_terminate)):
 #           for j in range(len(part_PR_codons_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
 #               if type(part_PR_codons_to_terminate[i][j]) == int:
 #                   codons_ = part_PR_codons_to_terminate[i][j]
 #                   PR_codons_to_terminate.append(codons_)
 #                   self.production_rules[i][j].append(codons_)
 #               else:
 #                   codons_ = sum(part_PR_codons_to_terminate[i][j])# - 1
 #                   PR_codons_to_terminate.append(codons_)
 #                   self.production_rules[i][j].append(codons_)
                    
        for i in range(len(self.production_rules)):
            part_PR_codons_to_terminate.append( list() )
            codons_non_terminal.append( list() )
            for j in range(len(self.production_rules[i])):
                part_PR_codons_to_terminate[i].append( list() )
                codons_non_terminal[i].append( list() )
                if self.production_rules[i][j][1] == 'terminal':
                    codons_non_terminal[i][j].append(None)
                    part_PR_codons_to_terminate[i][j] = 0 #part_PR_codons_to_terminate[i][j] represents the number of codons to terminate the remaining choices, then if this is a terminal, there are no remaining choices
                    if NT_codons_to_terminate[i] != 0:
                        NT_codons_to_terminate[i] = 0 #following the same idea from part_PR_codons_to_terminate[i][j]
                else:
                    for k in range(self.production_rules[i][j][2]): #arity
                        part_PR_codons_to_terminate[i][j].append( list() )
                        term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
                        codons_non_terminal[i][j].append('<' + term + '>')
        continue_ = True
        while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
            if None not in NT_codons_to_terminate:
                continue_ = False 
            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(codons_non_terminal[j][k])):
                            if self.non_terminals[i] == codons_non_terminal[j][k][l]:
                                if NT_codons_to_terminate[i] != None:
                                    if not part_PR_codons_to_terminate[j][k][l]:
                                        part_PR_codons_to_terminate[j][k][l] = NT_codons_to_terminate[i] + 1
                                        if [] not in part_PR_codons_to_terminate[j][k]:
                                            if not NT_codons_to_terminate[j]:
                                                NT_codons_to_terminate[j] = sum(part_PR_codons_to_terminate[j][k])
        PR_codons_to_terminate = []
        for i in range(len(part_PR_codons_to_terminate)):
            for j in range(len(part_PR_codons_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
                if type(part_PR_codons_to_terminate[i][j]) == int:
                    codons_ = part_PR_codons_to_terminate[i][j] + 1 #part_PR_codons_to_terminate[i][j] represents the number of codons to terminate the remaining choices, then we add 1 regarding the current choice
                    PR_codons_to_terminate.append(codons_)
                    self.production_rules[i][j].append(codons_)
                else:
                    codons_ = sum(part_PR_codons_to_terminate[i][j]) + 1 #part_PR_codons_to_terminate[i][j] represents the number of codons to terminate the remaining choices, then we add 1 regarding the current choice
                    PR_codons_to_terminate.append(codons_)
                    self.production_rules[i][j].append(codons_)
                    
        #New attributes
        self.max_codons_each_PR = []
        self.min_codons_each_PR = []
        for PR in self.production_rules:
            choices_ = []
            for choice in PR:
                choices_.append(choice[6])
            self.max_codons_each_PR.append(max(choices_))
            self.min_codons_each_PR.append(min(choices_))
        self.max_depth_each_PR = []
        self.min_depth_each_PR = []
        for PR in self.production_rules:
            choices_ = []
            for choice in PR:
                choices_.append(choice[5])
            self.max_depth_each_PR.append(max(choices_))
            self.min_depth_each_PR.append(min(choices_))
            
        self.initial_next_NT = re.search(r"\<(\w+)\>",self.start_rule).group()
        n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",self.start_rule)])
        self.initial_list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
        self.initial_list_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
        for term in re.findall(r"\<([\(\)\w,-.]+)\>",self.start_rule):
            NT_index = self.non_terminals.index('<' + term + '>')
            minimum_n_codons = []
            for PR in self.production_rules[NT_index]:
                minimum_n_codons.append(PR[6])
            self.initial_list_codons.append(min(minimum_n_codons))

def check_recursiveness(self, NT, stack):
    idx_NT = self.non_terminals.index(NT)
    for j in range(len(self.production_rules[idx_NT])):
        NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[idx_NT][j][0])
        NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
        unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
        recursive = False
  #      while unique_NTs.size and not recursive:
        for NT_to_check in unique_NTs:
            if NT_to_check in stack:
                recursive = True
                return recursive
            else:
                stack.append(NT_to_check) #Include the current NT to check it recursively
                recursive = check_recursiveness(self, NT_to_check, stack)
                if recursive:
                    return recursive
                stack.pop() #If the inclusion didn't show recursiveness, remove it before continuing
            #    return recursive
    return recursive#, stack

def mapper(genome, grammar, max_depth):
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])    
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_eager(genome, grammar, max_depth):
    """
    Identical to the previous one.
    TODO Solve the names later.
    """    

    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_lazy(genome, grammar, max_depth):
    """
    This mapper is similar to the previous one, but it does not consume codons
    when mapping a production rule with a single option."""
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            structure.append(index_production_chosen)
            idx_genome += 1
        
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
            
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def random_initialisation(ind_class, pop_size, bnf_grammar, min_init_genome_length, max_init_genome_length, max_init_depth, codon_size, codon_consumption, genome_representation):
        """
        
        """
        population = []
        
        for i in range(pop_size):
            genome = []
            init_genome_length = random.randint(min_init_genome_length, max_init_genome_length)
            for j in range(init_genome_length):
                genome.append(random.randint(0, codon_size))
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            population.append(ind)
    
        if genome_representation == 'list':
            return population
        else:
            raise ValueError("Unkonwn genome representation")

def sensible_initialisation(ind_class, pop_size, bnf_grammar, min_init_depth, 
                            max_init_depth, codon_size, codon_consumption,
                            genome_representation):
        """
        
        """
        
        if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
            tile_size = 0
            tile_idx = [] #Index of each grammar.production_rules in the tile
            tile_n_rules = [] #Number of choices (PRs) for each position of the tile
            for i in range(len(bnf_grammar.production_rules)):
                if len(bnf_grammar.production_rules[i]) == 1: #The PR has a single option
                    tile_idx.append(False)
                else:
                    tile_idx.append(tile_size)
                    tile_n_rules.append(len(bnf_grammar.production_rules[i]))
                    tile_size += 1            
        
        #Calculate the number of individuals to be generated with each method
        is_odd = pop_size % 2
        n_grow = int(pop_size/2)
        
        n_sets_grow = max_init_depth - min_init_depth + 1
        set_size = int(n_grow/n_sets_grow)
        remaining = n_grow % n_sets_grow
        
        n_full = n_grow + is_odd + remaining #if pop_size is odd, generate an extra ind with "full"
        
        #TODO check if it is possible to generate inds with max_init_depth
        
        population = []
        #Generate inds using "Grow"
        for i in range(n_sets_grow):
            max_init_depth_ = min_init_depth + i
            for j in range(set_size):
                remainders = [] #it will register the choices
                possible_choices = [] #it will register the respective possible choices
                if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                    PR_used_idx = [] #it will register the respective index of the PRs being used
    
                phenotype = bnf_grammar.start_rule
#                    remaining_NTs = [term for term in re.findall(r"(\<([\(\)\w,-.]+)\>|\<([\(\)\w,-.]+)\>\(\w+\))",phenotype)]
                #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
                depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
                idx_branch = 0 #index of the current branch being grown
                while len(remaining_NTs) != 0:
                    parameterised_ = False
                    idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                    total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                    actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth_]
                    Ch = random.choice(actual_options)
                    if codon_consumption == 'parameterised':#voltasen
                        if bnf_grammar.parameterised_non_terminals[idx_NT]:
                            parameterised_ = True
                    if parameterised_:
                        next_NT = re.search(remaining_NTs[0] + r"\(\S*\)", phenotype).group()
                        next_NT_parameterised = next_NT.split('>(')
                        next_NT_parameterised[0] = next_NT_parameterised[0] + '>'
                        next_NT_parameterised[1] = next_NT_parameterised[1][:-1] #remove the last parenthesis
                        #next_NT_parameterised[1] is the current level (integer number)
                        #grammar.parameterised_non_terminals[NT_index][1] is the parameter
                        exec(bnf_grammar.parameterised_non_terminals[idx_NT][1] + '=' + next_NT_parameterised[1])
                        PR_replace_ = Ch[0]
                        replace_levels_ = re.findall(r"\(\S*\)",PR_replace_)
                        for i in range(len(replace_levels_)):
                            level_ = eval(replace_levels_[i])
                            PR_replace_ = PR_replace_.replace(replace_levels_[i], '(' + str(level_) + ')')
                        phenotype = phenotype.replace(next_NT, PR_replace_, 1)
                    else:
                        phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                    depths[idx_branch] += 1
                    if codon_consumption == 'eager' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
                        remainders.append(Ch[3])
                        possible_choices.append(len(total_options))
                    elif codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                        if len(total_options) > 1:
                            remainders.append(Ch[3])
                            possible_choices.append(len(total_options))
                            if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                                PR_used_idx.append(idx_NT)                       
                    
                    if Ch[2] > 1:
                        if idx_branch == 0:
                            depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        else:
                            depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    if Ch[1] == 'terminal':
                        idx_branch += 1
                    
                    #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
                    remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
                
                #Generate the genome
                genome = []
                if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
                    for k in range(len(remainders)):
                        codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                        genome.append(codon)
                elif codon_consumption == 'leap':
                    for k in range(len(remainders)):
                        codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                        for l in range(tile_size):
                            if l == tile_idx[PR_used_idx[k]]:#volta
                                genome.append(codon)
                            else:
                                genome.append(random.randint(0, codon_size))
                elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
                    #Firstly we need to know how many tiles we will have
                    tile_map = [[False]*tile_size]
                    n_tiles = 0
                    order_map_inside_tile = 1 #The first one to be used will receive the order 1
                    for k in range(len(remainders)):
                        for l in range(tile_size):
                            if l == tile_idx[PR_used_idx[k]]: 
                                if tile_map[n_tiles][l]: #If we already used this position, we open the next tile
                                    order_map_inside_tile = 1 #To check how position is being mapped firstly inside the tile
                                    n_tiles += 1
                                    tile_map.append([False]*tile_size)
                                    tile_map[n_tiles][l] = order_map_inside_tile
                                    order_map_inside_tile += 1
                                else: #If not, we keep in the same tile
                                    tile_map[n_tiles][l] = order_map_inside_tile
                                    order_map_inside_tile += 1
                    #Now we know how the tiles are distributed, so we can map
                    positions_used_each_tile = []
                    for k in range(len(tile_map)):
                        positions = 0
                        for l in range(tile_size):
                            if tile_map[k][l]:
                                positions += 1
                        positions_used_each_tile.append(positions)    
                        
                    id_mapping = 0
                    
                    for k in range(len(tile_map)):
                        for l in range(tile_size):
                            if tile_map[k][l]:
                                if codon_consumption == 'leap2':
                                    codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[id_mapping+tile_map[k][l]-1])) * possible_choices[id_mapping+tile_map[k][l]-1]) + remainders[id_mapping+tile_map[k][l]-1]
                                elif codon_consumption == 'leap3':
                                    codon = remainders[id_mapping+tile_map[k][l]-1]#possible_choices[id_mapping+tile_map[k][l]-1]                                    
                                genome.append(codon)
                                #id_mapping += 1
                            else:
                                if codon_consumption == 'leap2':
                                    genome.append(random.randint(0, codon_size))
                                elif codon_consumption == 'leap3':
                                    genome.append(random.randint(0, tile_n_rules[l]))
#                                print(genome)
                        id_mapping += positions_used_each_tile[k]
                    
#                    order_map_inside_tile = 0
#                    for k in range(len(remainders)):
#                        codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
#                        for l in range(tile_size):
#                            if tile_map[n_tiles][l]: #If we already used this position, we open the next tile
#                                if tile_map[n_tiles][l] == order_map_inside_tile    
#                            n_tiles += 1
#                                tile_map.append([False]*tile_size)
#                                tile_map[n_tiles][l] == True
#                            else: #If not, we keep in the same tile
#                                if l == tile_idx[PR_used_idx[k]]: 
#                                    tile_map[n_tiles][l] == True
                else:
                    raise ValueError("Unknown mapper")
                    
                #Include a tail with 50% of the genome's size
                size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                for k in range(size_tail):
                    genome.append(random.randint(0,codon_size))
                    
                #Initialise the individual and include in the population
                ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption)
                
   #             if ind.depth > 5:
   #                 print(ind.phenotype)
   #                 print(ind.depth)
   #                 input("enter")
   #             print(phenotype)
   #             print(remainders)
   #             print(ind.structure)
   #             print(ind.invalid)
#                input()
                
                #Check if the individual was mapped correctly
                if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                    raise Exception('error in the mapping')
                    
                population.append(ind)    

            
        for i in range(n_full):
            remainders = [] #it will register the choices
            possible_choices = [] #it will register the respective possible choices
            if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                PR_used_idx = [] #it will register the respective index of the PRs being used

            phenotype = bnf_grammar.start_rule
            #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
            depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
            idx_branch = 0 #index of the current branch being grown

            while len(remaining_NTs) != 0:
                parameterised_ = False
                idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth]
                recursive_options = [PR for PR in actual_options if PR[4]]
                if len(recursive_options) > 0:
                    Ch = random.choice(recursive_options)
                else:
                    Ch = random.choice(actual_options)
                if codon_consumption == 'parameterised':#voltasen
                    if bnf_grammar.parameterised_non_terminals[idx_NT]:
                        parameterised_ = True
                if parameterised_:
                    next_NT = re.search(remaining_NTs[0] + r"\(\S*\)", phenotype).group()
                    next_NT_parameterised = next_NT.split('>(')
                    next_NT_parameterised[0] = next_NT_parameterised[0] + '>'
                    next_NT_parameterised[1] = next_NT_parameterised[1][:-1] #remove the last parenthesis
                    #next_NT_parameterised[1] is the current level (integer number)
                    #grammar.parameterised_non_terminals[NT_index][1] is the parameter
                    exec(bnf_grammar.parameterised_non_terminals[idx_NT][1] + '=' + next_NT_parameterised[1])
                    #PR_replace_ = bnf_grammar.production_rules[idx_NT][Ch[0]][0]
                    PR_replace_ = Ch[0]
                    replace_levels_ = re.findall(r"\(\S*\)",PR_replace_)
                    for i in range(len(replace_levels_)):
                        level_ = eval(replace_levels_[i])
                        PR_replace_ = PR_replace_.replace(replace_levels_[i], '(' + str(level_) + ')')
                    phenotype = phenotype.replace(next_NT, PR_replace_, 1)
                else:
                    phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                depths[idx_branch] += 1
                if codon_consumption == 'eager' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
                    remainders.append(Ch[3])
                    possible_choices.append(len(total_options))
                elif codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                    if len(total_options) > 1:
                        remainders.append(Ch[3])
                        possible_choices.append(len(total_options))
                        if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                            PR_used_idx.append(idx_NT)       
                
                if Ch[2] > 1:
                    if idx_branch == 0:
                        depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    else:
                        depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                if Ch[1] == 'terminal':
                    idx_branch += 1
                
                #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
            
            #Generate the genome
            genome = []
            if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
            	for j in range(len(remainders)):
            		codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
            		genome.append(codon)
            elif codon_consumption == 'leap':
            	for j in range(len(remainders)):
            		codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
            		for k in range(tile_size):
            			if k == tile_idx[PR_used_idx[j]]:
            				genome.append(codon)
            			else:
            				genome.append(random.randint(0, codon_size))
            elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
                #Firstly we need to know how many tiles we will have
                tile_map = [[False]*tile_size]
                n_tiles = 0
                order_map_inside_tile = 1 #The first one to be used will receive the order 1
                for k in range(len(remainders)):
                    for l in range(tile_size):
                        if l == tile_idx[PR_used_idx[k]]: 
                            if tile_map[n_tiles][l]: #If we already used this position, we open the next tile
                                order_map_inside_tile = 1 #To check how position is being mapped firstly inside the tile
                                n_tiles += 1
                                tile_map.append([False]*tile_size)
                                tile_map[n_tiles][l] = order_map_inside_tile
                                order_map_inside_tile += 1
                            else: #If not, we keep in the same tile
                                tile_map[n_tiles][l] = order_map_inside_tile
                                order_map_inside_tile += 1
                #Now we know how the tiles are distributed, so we can map
                positions_used_each_tile = []
                for k in range(len(tile_map)):
                    positions = 0
                    for l in range(tile_size):
                        if tile_map[k][l]:
                            positions += 1
                    positions_used_each_tile.append(positions)    
                    
                id_mapping = 0
                
                for k in range(len(tile_map)):
                    for l in range(tile_size):
                        if tile_map[k][l]:
                            if codon_consumption == 'leap2':
                                codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[id_mapping+tile_map[k][l]-1])) * possible_choices[id_mapping+tile_map[k][l]-1]) + remainders[id_mapping+tile_map[k][l]-1]
                            elif codon_consumption == 'leap3':
                                codon = remainders[id_mapping+tile_map[k][l]-1]#possible_choices[id_mapping+tile_map[k][l]-1]
                            genome.append(codon)
                            #id_mapping += 1
                        else:
                            if codon_consumption == 'leap2':
                                genome.append(random.randint(0, codon_size))
                            elif codon_consumption == 'leap3':
                                genome.append(random.randint(0, tile_n_rules[l]))
                    id_mapping += positions_used_each_tile[k]
            else:
            	raise ValueError("Unknown mapper")

            #Include a tail with 50% of the genome's size
            if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
                size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
            elif codon_consumption == 'leap':
                raise ValueError("This mapping process was not implemented for this initialisation method")
            elif codon_consumption == 'leap2':
                n_tiles_tail = max(int(0.5*n_tiles), 1)
                size_tail = n_tiles_tail * tile_size
            elif codon_consumption == 'leap3':
                raise ValueError("This mapping process was not implemented for this initialisation method")
            
            for j in range(size_tail):
                genome.append(random.randint(0,codon_size))
                
            #Initialise the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            
            #Check if the individual was mapped correctly
            if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                raise Exception('error in the mapping')

            population.append(ind)    
    
        if genome_representation == 'list':
            return population
        elif genome_representation == 'numpy':
            for ind in population:
                ind.genome = np.array(ind.genome)
            return population
        else:
            raise ValueError("Unkonwn genome representation")

def crossover_onepoint(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       invalidate_max_depth,
                       genome_representation='list', max_genome_length=None):
    """
    
    """
    if max_genome_length:
        raise ValueError("max_genome_length not implemented in this onepoint")
    
    if parent0.invalid: #used_codons = 0
        possible_crossover_codons0 = len(parent0.genome)
    else:
        possible_crossover_codons0 = min(len(parent0.genome), parent0.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_codons1 = len(parent1.genome)
    else:
        possible_crossover_codons1 = min(len(parent1.genome), parent1.used_codons)
#        print()
    
    parent0_genome = parent0.genome.copy()
    parent1_genome = parent1.genome.copy()
    continue_ = True
#    a = 0
    while continue_:
        #Set points for crossover within the effective part of the genomes
        point0 = random.randint(1, possible_crossover_codons0)
        point1 = random.randint(1, possible_crossover_codons1)
        
        if genome_representation == 'list':
            #Operate crossover
            new_genome0 = parent0_genome[0:point0] + parent1_genome[point1:]
            new_genome1 = parent1_genome[0:point1] + parent0_genome[point0:]
        else:
            raise ValueError("Only 'list' representation is implemented")
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
        
        if invalidate_max_depth: # In the mapping, if a ind surpasses max depth, it is invalid, and we won't redo crossover
            continue_ = False
        else: # We check if a ind surpasses max depth, and if so we will redo crossover
            continue_ = new_ind0.depth > max_depth or new_ind1.depth > max_depth

        

        
#        if not check_:
#            print()
#            print("checking")
#            print("parent0:")
#            print("length = ", len(parent0_genome))
#            print("used codons = ", parent0.used_codons)
#            print("invalid = ", parent0.invalid)
#            print("cut point = ", point0)
#            print("parent1:")
#            print("length = ", len(parent1_genome))
#            print("used codons = ", parent1.used_codons)
#            print("invalid = ", parent1.invalid)
#            print("cut point = ", point1)
#            check = True
            
        #if len(new_genome0) == 1 or len(new_genome1) == 1:
            #print(continue_)
#            if continue_:
#                print()
#                print("parent0:")
#                print("length = ", len(parent0_genome))
#                print("used codons = ", parent0.used_codons)
#                print("invalid = ", parent0.invalid)
#                print("cut point = ", point0)
#                print("parent1:")
#                print("length = ", len(parent1_genome))
#                print("used codons = ", parent1.used_codons)
#                print("invalid = ", parent1.invalid)
#                print("cut point = ", point1)
#                check_ = False
                
#        if not continue_:
#            if len(new_genome0) == 1 or len(new_genome1) == 1:
#                print()
#                print("stopped")
#                print("parent0:")
#                print("length = ", len(parent0_genome))
#                print("used codons = ", parent0.used_codons)
#                print("invalid = ", parent0.invalid)
#                print("cut point = ", point0)
#                print("parent1:")
#                print("length = ", len(parent1_genome))
#                print("used codons = ", parent1.used_codons)
#                print("invalid = ", parent1.invalid)
#                print("cut point = ", point1)
                      
        
    del new_ind0.fitness.values, new_ind1.fitness.values
    return new_ind0, new_ind1   
    
def mutation_int_flip_per_codon(ind, mut_probability, codon_size, bnf_grammar, max_depth, 
                                codon_consumption, invalidate_max_depth,
                                max_genome_length):
    """

    """
    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    continue_ = True
    
    genome = copy.deepcopy(ind.genome)
    mutated_ = False

    while continue_:
        for i in range(possible_mutation_codons):
            if random.random() < mut_probability:
                genome[i] = random.randint(0, codon_size)
                mutated_ = True
               # break
    
        new_ind = reMap(ind, genome, bnf_grammar, max_depth, codon_consumption)
        
        if invalidate_max_depth: # In the mapping, if ind surpasses max depth, it is invalid, and we won't redo mutation
            continue_ = False
        else: # We check if ind surpasses max depth, and if so we will redo mutation
            continue_ = new_ind.depth > max_depth
        
#    if max_genome_length:
#        if new_ind.depth > max_depth or len(new_ind.genome) > max_genome_length:
#            return ind,
#        else:
#            return new_ind,
#    else:
        #if new_ind.depth > max_depth:
        #    return ind,
        #else:
    if mutated_:
        del new_ind.fitness.values
    return new_ind,
        
def reMap(ind, genome, bnf_grammar, max_tree_depth, codon_consumption):
    #TODO refazer todo o reMap para nao copiar o ind
    #
    #ind = Individual(genome, bnf_grammar, max_tree_depth, codon_consumption)
    #ind = Individual(genome, bnf_grammar, max_tree_depth, codon_consumption)
    ind.genome = genome
    if codon_consumption == 'lazy':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_lazy(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_eager(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'leap':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.tile_size, ind.effective_positions = mapper_leap(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.tile_size, ind.effective_positions = mapper_leap2(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'parameterised':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_parameterised(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'multiGE':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_multi(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'multichromosomalGE':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_multichromosomal(genome, bnf_grammar, max_tree_depth)         
    elif codon_consumption == 'cosmo_eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.cosmo = mapper_cosmo(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'cosmo_eager_depth':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.cosmo = mapper_cosmo_ext(genome, bnf_grammar, max_tree_depth)    
    elif codon_consumption == 'cosmo_total':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.cosmo = mapper_cosmo_total(genome, bnf_grammar, max_tree_depth)     
    else:
        raise ValueError("Unknown mapper")
        
    return ind

def replace_nth(string, substring, new_substring, nth):
    find = string.find(substring)
    i = find != -1
    while find != -1 and i != nth:
        find = string.find(substring, find + 1)
        i += 1
    if i == nth:
        return string[:find] + new_substring + string[find+len(substring):]
    return string