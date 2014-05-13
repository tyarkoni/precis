import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
import copy
import logging
import random
from scythe import abbreviate
from scythe import evaluate
from scythe import plot
from scythe.base import AbbreviatedMeasure


class Generator:

    def __init__(self, abbreviator=None, evaluator=None, cross_validate=False, **kwargs):
        '''
        Args:
            abbreviator: The KeyGenerator to use to generate a scoring key from the data.
                If None, uses the top N absolute correlation approach in Yarkoni (2010).
            evaluator: The LossFunction to minimize using the GA. If None, uses the 
                loss function in Yarkoni (2010).
            cross_validate: Whether or not to use split-half cross-validation.
            kwargs: Optional arguments to pass to DEAP.
        '''
        self.cross_val = cross_validate

        if abbreviator is None:
            abbreviator = abbreviate.TopNAbbreviator()
        self.abbreviator = abbreviator

        if evaluator is None:
            evaluator = evaluate.YarkoniEvaluator()
        self.evaluator = evaluator

        # Deap settings
        self.zero_to_one_ratio = kwargs.get('zero_to_one_ratio', 0.5)
        self.indpb = kwargs.get('indpb', 0.05)
        self.tourn_size = kwargs.get('tourn_size', 3)
        self.pop_size = kwargs.get('pop_size', 200)
        self.cxpb = kwargs.get('cxpb', 0.8)
        self.mutpb = kwargs.get('mutpb', 0.2)

        self.logbook = tools.Logbook()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", min)
        self.stats = stats


    def _random_boolean(self, zero_to_one_ratio):
        return random.random() < zero_to_one_ratio


    def run(self, measure, n_gens=100, seed=None, **kwargs):
        ''' Main abbreviated measure generation function.

        Args:
            measure: A Measure instance to abbreviate
            n_gens: Number of generations to run GA for
            seed: Optional integer to use as random seed
            kwargs: Additional keywords to pass on to the evaluation method
                of the current LossFunction class.

        Returns: A list of items included in the abbreviated measure.
        '''

        # Set random seed for both native Python and Numpy, to be safe
        random.seed(seed)
        np.random.seed(seed)

        # Set up the GA
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", self._random_boolean, self.zero_to_one_ratio)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
            toolbox.attr_bool, measure.n_X)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.indpb)
        toolbox.register("select", tools.selTournament, tournsize=self.tourn_size)

        # Initialize population
        pop = toolbox.population(n=self.pop_size)

        self.measure = measure
        self.evaluation_keywords = kwargs

        # Cross-validation
        if self.cross_val:
            inds = range(self.measure.n_subjects)
            random.shuffle(inds)
            self.train_subs = [x for i, x in enumerate(inds) if i % 2 != 0]
            self.test_subs = [x for x in inds if x not in self.train_subs]
            self.test_measure = copy.deepcopy(self.measure)
            self.measure.select_subjects(self.train_subs)
            self.test_measure.select_subjects(self.test_subs)
        
        self.evolve(pop, toolbox, n_gens, cxpb=self.cxpb, mutpb=self.mutpb)
        final_items = self.best_individuals[-1]

        # If cross-validation was used, activate the hold-out subjects
        measure = self.test_measure if self.cross_val else self.measure
        self.best = AbbreviatedMeasure(measure, final_items, self.abbreviator, self.evaluator, stats=True)    
        return self.best


    def evolve(self, population, toolbox, ngen, cxpb, mutpb, verbose=True):
        ''' Main evolution algorithm. A tweaked version of the eaSimple algorithm included in 
        the DEAP package that adds per-generation logging of the best individual's properties
        and drops all the Statistics/HallOfFame stuff (since we're handling that ourselves).
        See DEAP documentation of algorithms.eaSimple() for all arguments.
        '''

        # if verbose:
        #     column_names = ["gen", "evals"]
        #     if stats is not None:
        #         column_names += stats.functions.keys()
            # logger = tools.Logbook(column_names)
            # logger.logHeader()
            # logger.logGeneration(evals=len(population), gen=0, stats=stats)

        # Store best individual in each generation, and associated measure
        self.best_individuals = []
        self.best_measures = []

        # Begin the generational process
        for gen in range(0, ngen):

            # Select the next generation individuals
            offspring = toolbox.select(population, k=len(population))
                
            # Variate the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            # Replace the current population by the offspring
            offspring = sorted(offspring, key=lambda x: x.fitness, reverse = True)
            population[:] = offspring

            # Save best individual as an AbbreviatedMeasure
            self.best_individuals.append(population[0])
            best_abb = AbbreviatedMeasure(self.measure, population[0], self.abbreviator, self.evaluator, stats=True) 
            self.best_measures.append(best_abb)

            # Update the statistics with the new population
            if self.stats is not None:
                record = self.stats.compile(population)
                self.logbook.record(gen=gen, evals=len(population), **record)

            # if verbose:
                # logger.logGeneration(evals=len(invalid_ind), gen=gen, stats=stats)


    def evaluate(self, individual):
        m = self.abbreviator.abbreviate_apply(self.measure.dataset, select=individual)
        loss = self.evaluator.evaluate(m, **self.evaluation_keywords)
        return (loss, )


    def save(self):
        ''' Save results of abbreviation. '''
        pass


    def plot_history(self, **kwargs):
        ''' Convenience wrapper for history() in plot module. '''
        return plot.history(self, **kwargs)

