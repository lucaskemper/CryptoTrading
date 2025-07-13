#!/usr/bin/env python3
"""
Strategy Optimization System
Optimize trading strategy parameters using grid search and genetic algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import itertools
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    parameters: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    fitness_score: float

class StrategyOptimizer:
    """Optimize trading strategy parameters"""
    
    def __init__(self, backtest_engine, parameter_ranges: Dict[str, List]):
        self.backtest_engine = backtest_engine
        self.parameter_ranges = parameter_ranges
        self.results = []
        
    def grid_search(self, max_combinations: int = 1000) -> List[OptimizationResult]:
        """Perform grid search optimization"""
        logger.info("Starting grid search optimization...")
        
        # Generate all parameter combinations
        param_names = list(self.parameter_ranges.keys())
        param_values = list(self.parameter_ranges.values())
        
        combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(combinations) > max_combinations:
            # Sample combinations randomly
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]
        
        logger.info(f"Testing {len(combinations)} parameter combinations...")
        
        # Test each combination
        results = []
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            try:
                result = self._test_parameters(params)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(combinations)} tests")
                    
            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue
        
        # Sort by fitness score
        results.sort(key=lambda x: x.fitness_score, reverse=True)
        self.results = results
        
        logger.info(f"Grid search completed. Best fitness: {results[0].fitness_score:.3f}")
        return results
    
    def genetic_algorithm(self, population_size: int = 50, generations: int = 20) -> List[OptimizationResult]:
        """Perform genetic algorithm optimization"""
        logger.info("Starting genetic algorithm optimization...")
        
        # Initialize population
        population = self._initialize_population(population_size)
        
        best_results = []
        
        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")
            
            # Evaluate population
            results = []
            for params in population:
                try:
                    result = self._test_parameters(params)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error testing parameters: {e}")
                    continue
            
            # Sort by fitness
            results.sort(key=lambda x: x.fitness_score, reverse=True)
            best_results.extend(results[:5])  # Keep top 5 from each generation
            
            # Selection and crossover
            if generation < generations - 1:
                population = self._evolve_population(results, population_size)
        
        # Sort final results
        best_results.sort(key=lambda x: x.fitness_score, reverse=True)
        self.results = best_results
        
        logger.info(f"Genetic algorithm completed. Best fitness: {best_results[0].fitness_score:.3f}")
        return best_results
    
    def _initialize_population(self, size: int) -> List[Dict]:
        """Initialize random population for genetic algorithm"""
        population = []
        
        for _ in range(size):
            params = {}
            for param_name, param_range in self.parameter_ranges.items():
                if isinstance(param_range[0], int):
                    # Integer parameter
                    params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    # Float parameter
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
            
            population.append(params)
        
        return population
    
    def _evolve_population(self, results: List[OptimizationResult], size: int) -> List[Dict]:
        """Evolve population using selection, crossover, and mutation"""
        # Selection: keep top 20%
        elite_size = max(1, size // 5)
        elite = [result.parameters for result in results[:elite_size]]
        
        # Tournament selection for rest
        new_population = elite.copy()
        
        while len(new_population) < size:
            # Tournament selection
            tournament_size = 3
            tournament = np.random.choice(len(results), tournament_size, replace=False)
            winner = max(tournament, key=lambda i: results[i].fitness_score)
            
            # Crossover with elite member
            if elite:
                parent1 = results[winner].parameters
                parent2 = np.random.choice(elite)
                child = self._crossover(parent1, parent2)
            else:
                child = results[winner].parameters.copy()
            
            # Mutation
            child = self._mutate(child)
            new_population.append(child)
        
        return new_population
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two parameter sets"""
        child = {}
        
        for param_name in parent1.keys():
            if np.random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _mutate(self, params: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutate parameters"""
        mutated = params.copy()
        
        for param_name, value in mutated.items():
            if np.random.random() < mutation_rate:
                param_range = self.parameter_ranges[param_name]
                
                if isinstance(value, int):
                    # Integer mutation
                    mutated[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    # Float mutation
                    mutated[param_name] = np.random.uniform(param_range[0], param_range[1])
        
        return mutated
    
    def _test_parameters(self, params: Dict) -> OptimizationResult:
        """Test a parameter combination"""
        # Update backtest engine with new parameters
        self.backtest_engine.update_parameters(params)
        
        # Run backtest
        results = self.backtest_engine.run_backtest()
        
        # Calculate fitness score
        fitness_score = self._calculate_fitness(results)
        
        return OptimizationResult(
            parameters=params,
            total_return=results.get('total_return', 0.0),
            sharpe_ratio=results.get('sharpe_ratio', 0.0),
            max_drawdown=results.get('max_drawdown', 0.0),
            win_rate=results.get('win_rate', 0.0),
            total_trades=results.get('total_trades', 0),
            fitness_score=fitness_score
        )
    
    def _calculate_fitness(self, results: Dict) -> float:
        """Calculate fitness score for optimization"""
        # Weighted combination of metrics
        total_return = results.get('total_return', 0.0)
        sharpe_ratio = results.get('sharpe_ratio', 0.0)
        max_drawdown = abs(results.get('max_drawdown', 0.0))
        win_rate = results.get('win_rate', 0.0)
        total_trades = results.get('total_trades', 0)
        
        # Penalize low trade count
        trade_penalty = max(0, 1 - total_trades / 100)
        
        # Calculate fitness
        fitness = (
            0.4 * total_return +           # 40% weight on returns
            0.3 * sharpe_ratio +           # 30% weight on risk-adjusted returns
            0.2 * win_rate +               # 20% weight on win rate
            0.1 * (1 - max_drawdown) -     # 10% weight on drawdown (penalty)
            trade_penalty                   # Penalty for insufficient trades
        )
        
        return fitness
    
    def get_best_parameters(self, top_n: int = 5) -> List[OptimizationResult]:
        """Get top N parameter combinations"""
        if not self.results:
            return []
        
        return self.results[:top_n]
    
    def save_results(self, filename: str):
        """Save optimization results to file"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'parameters': result.parameters,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'fitness_score': result.fitness_score
            })
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def load_results(self, filename: str) -> List[OptimizationResult]:
        """Load optimization results from file"""
        if not os.path.exists(filename):
            logger.error(f"File not found: {filename}")
            return []
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            result = OptimizationResult(
                parameters=item['parameters'],
                total_return=item['total_return'],
                sharpe_ratio=item['sharpe_ratio'],
                max_drawdown=item['max_drawdown'],
                win_rate=item['win_rate'],
                total_trades=item['total_trades'],
                fitness_score=item['fitness_score']
            )
            results.append(result)
        
        self.results = results
        logger.info(f"Loaded {len(results)} results from {filename}")
        return results

# Example usage
if __name__ == "__main__":
    # Define parameter ranges for optimization
    parameter_ranges = {
        'z_score_threshold': [1.0, 1.5, 2.0, 2.5, 3.0],
        'correlation_threshold': [0.3, 0.5, 0.7, 0.8, 0.9],
        'position_size_limit': [0.01, 0.02, 0.05, 0.1, 0.2],
        'stop_loss_pct': [0.02, 0.05, 0.1, 0.15, 0.2],
        'take_profit_pct': [0.05, 0.1, 0.2, 0.3, 0.4]
    }
    
    # Initialize optimizer (you would need to pass your backtest engine)
    # optimizer = StrategyOptimizer(backtest_engine, parameter_ranges)
    
    # Run grid search
    # results = optimizer.grid_search(max_combinations=100)
    
    # Run genetic algorithm
    # results = optimizer.genetic_algorithm(population_size=30, generations=10)
    
    # Get best parameters
    # best_params = optimizer.get_best_parameters(top_n=5)
    # for i, result in enumerate(best_params):
    #     print(f"Rank {i+1}: Fitness={result.fitness_score:.3f}, "
    #           f"Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.3f}")
    
    print("Strategy optimizer ready for use!") 