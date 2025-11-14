import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling, BinaryRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


class FARCHDProblem(Problem):
    """
    FARCHD problem with mixed chromosome:
    - Binary part for rule selection (CS)
    - Real part for partition tuning (CT)
    """
    def __init__(self, rules, partitions, x, y, delta):
        self.rules = rules
        self.partitions = partitions
        self.x = x
        self.y = y
        self.delta = delta

        # Binary variables for rule selection
        self.n_rules = len(rules)

        # Real variables for partition tuning
        # Calculate total partition variables - always treat as list structure
        self.n_partition_vars = sum(len(var_partitions) for var_partitions in partitions)

        # Store partition structure for later use
        self.partition_structure = []
        for var_partitions in partitions:
            self.partition_structure.append(len(var_partitions))

        # Define bounds
        xl_binary = np.zeros(self.n_rules)  # Binary part bounds
        xu_binary = np.ones(self.n_rules)

        xl_real = np.full(self.n_partition_vars, -0.5)  # Real part bounds
        xu_real = np.full(self.n_partition_vars, 0.5)

        # Combine bounds
        xl = np.concatenate([xl_binary, xl_real])
        xu = np.concatenate([xu_binary, xu_real])

        super().__init__(
            n_var=self.n_rules + self.n_partition_vars,
            n_obj=1,  # Single objective: maximize fitness
            n_constr=0,
            xl=xl,
            xu=xu,
            type_var=np.concatenate([
                np.full(self.n_rules, bool),
                np.full(self.n_partition_vars, float)
            ])
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Split chromosome into binary (rule selection) and real (partition tuning) parts
        binary_part = X[:, :self.n_rules]
        real_part = X[:, self.n_rules:]

        fitnesses = []
        for i in range(len(X)):
            chromosome = [binary_part[i], real_part[i]]
            fitness = self._chromosome_evaluation(chromosome)
            # Pymoo minimizes, so we negate for maximization
            fitnesses.append(-fitness)

        out["F"] = np.array(fitnesses).reshape(-1, 1)

    def _chromosome_evaluation(self, chromosome):
        """Evaluate a single chromosome (adapted from your original function)"""
        N = len(self.y)
        active_rules = np.array(self.rules, dtype=object)[chromosome[0] == 1]

        if active_rules.size == 0:
            return 0

        symbolic_translation = chromosome[1]
        translated_partitions = self._reshape_translation(symbolic_translation)

        y_out = self._inference(self.x, active_rules, translated_partitions)
        Hits = sum(y_out == self.y)

        NRinitial = len(self.rules)
        NR = len(active_rules)
        Fitness = Hits/N - self.delta * NRinitial/(NRinitial - NR + 1.0)

        return Fitness

    def _reshape_translation(self, symbolic_translation):
        """Reshape the translation vector according to partition structure"""
        # Variable structure - handle different number of labels per variable
        translated_partitions = []
        idx = 0
        for var_idx, n_labels in enumerate(self.partition_structure):
            var_translation = symbolic_translation[idx:idx+n_labels]
            var_partitions = self.partitions[var_idx]
            # Add translation to each partition of this variable
            translated_var = []
            for label_idx, partition in enumerate(var_partitions):
                translated_partition = partition + var_translation[label_idx]
                translated_var.append(translated_partition)
            translated_partitions.append(translated_var)
            idx += n_labels
        return translated_partitions

    def _inference(self, x, rules, partitions):
        """Inference function (adapted from your original)"""
        n = len(x)
        best_membership = np.zeros(n)
        class_out = np.full((n), rules[0].class_j)
        mshp = np.ones(n)

        for rule in rules:
            variables, labels = rule.variables, rule.labels
            membership = mshp.copy()
            for var, label in zip(variables, labels):
                membership_function = partitions[var][label]
                membership *= np.interp(x[:, var], membership_function, [0, 1, 0], left=0, right=0)

            best_idx = membership > best_membership
            class_out = np.where(best_idx, rule.class_j, class_out)
            best_membership = np.where(best_idx, membership, best_membership)

        return class_out


class FARCHDCrossover(Crossover):
    """
    Crossover for FARCHD mixed chromosome:
    - Uniform crossover for binary part (rule selection)
    - SBX for real part (partition tuning)
    """
    def __init__(self, n_rules, prob_binary=0.9, prob_real=0.9, eta=15):
        super().__init__(2, 2)  # 2 parents -> 2 offspring
        self.n_rules = n_rules
        self.binary_crossover = UniformCrossover(prob=prob_binary)
        self.real_crossover = SBX(prob=prob_real, eta=eta)

    def _do(self, problem, X, **kwargs):
        offspring = np.empty_like(X)

        # Split into binary and real parts
        binary_part = X[:, :self.n_rules]
        real_part = X[:, self.n_rules:]

        # Apply crossover operations
        offspring[:, :self.n_rules] = self.binary_crossover._do(problem, binary_part, **kwargs)
        offspring[:, self.n_rules:] = self.real_crossover._do(problem, real_part, **kwargs)

        return offspring


class FARCHDMutation(Mutation):
    """
    Mutation for FARCHD mixed chromosome:
    - Bit flip mutation for binary part
    - Polynomial mutation for real part
    """
    def __init__(self, n_rules, prob_binary=0.1, prob_real=0.1, eta=20):
        super().__init__()
        self.n_rules = n_rules
        self.binary_mutation = BitflipMutation(prob=prob_binary)
        self.real_mutation = PM(prob=prob_real, eta=eta)

    def _do(self, problem, X, **kwargs):
        # Split into binary and real parts
        binary_part = X[:, :self.n_rules]
        real_part = X[:, self.n_rules:]

        # Apply mutations
        binary_mutated = self.binary_mutation._do(problem, binary_part, **kwargs)
        real_mutated = self.real_mutation._do(problem, real_part, **kwargs)

        # Combine mutated parts
        mutated = np.concatenate([binary_mutated, real_mutated], axis=1)
        return mutated


class FARCHDSampling(Sampling):
    """
    Custom sampling for FARCHD mixed chromosome
    """
    def __init__(self, rules, partitions):
        super().__init__()
        self.n_rules = len(rules)

        # Calculate partition variables based on actual structure
        self.n_partition_vars = sum(len(var_partitions) for var_partitions in partitions)

        self.binary_sampling = BinaryRandomSampling()
        self.real_sampling = FloatRandomSampling()

    def _do(self, problem, n_samples, **kwargs):
        # Create the first individual as all rules selected with zero tuning
        samples = np.zeros((n_samples, self.n_rules + self.n_partition_vars))

        # First sample: all rules selected, no tuning
        samples[0, :self.n_rules] = 1  # All rules selected
        samples[0, self.n_rules:] = 0  # No partition tuning

        # Generate remaining samples randomly
        for i in range(1, n_samples):
            # Random binary part (rule selection)
            samples[i, :self.n_rules] = np.random.randint(2, size=self.n_rules)
            # Random real part (partition tuning)
            samples[i, self.n_rules:] = np.random.uniform(-0.5, 0.5, size=self.n_partition_vars)

        return samples


# Example usage
if __name__ == "__main__":
    # Define problem parameters
    rules = [...]  # Your rule set here
    partitions = [...]  # Your partition set here
    x = np.random.rand(100, 5)  # Example input data
    y = np.random.randint(0, 2, 100)  # Example binary target data
    delta = 0.1  # Example delta value for fitness calculation

    # Create problem
    problem = FARCHDProblem(rules, partitions, x, y, delta)

    # Create custom operators
    sampling = FARCHDSampling(rules, partitions)
    crossover = FARCHDCrossover(len(rules))
    mutation = FARCHDMutation(len(rules))

    # Create algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 50),
        verbose=True,
        seed=1
    )

    # Display results
    print(f"Number of solutions: {len(res.X)}")
    print(f"First solution (rules): {res.X[0][:len(rules)]}")
    print(f"First solution (partitions): {res.X[0][len(rules):]}")
    print(f"First solution objective: {res.F[0]}")
