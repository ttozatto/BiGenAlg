"""
Genetic Algorithm for binary variables.

Parameters
-----------
  cost_f : cost function, must have a list as parameter and return the cost.
  nvars : number of binary variables.
  gene_size : size of each variable.
  pop_size : population size. number of chromosomes in each generation.
  x_rate : rate of population that will continue in the next generation.
  mutation_rate : rate of population that will suffer mutation.
  tolerance : minimum change in cost to converge.
  n_iter : maximum number of iterations.

Example
-------
  >>> def f(c):
  >>>   return 300*c[0] - 100*c[1] + 200*c[2] - 500*c[3]
  >>> galg = GA(f, nvars=4, pop_size=20, tolerance = 1e-4,
  >>>           patience=5, max_iter=50)
  >>> galg.run()
  Tolerance reached!

  Number of iterations: 9
  Best chromosome: [0 1 0 1]
  Cost: -600
"""
import numpy as np

class GA:
  def __init__(self, cost_f, nvars, gene_size=1, pop_size=8, x_rate=0.5,
               mutation_rate=0.2, tolerance=None, patience=5, max_iter=1e4):
    """
    Genetic Algorithm for binary variables.

    Parameters
    -----------
      cost_f : cost function, must have a list as parameter and return the cost.
      nvars : number of binary variables.
      gene_size : size of each variable.
      pop_size : population size. number of chromosomes in each generation.
      x_rate : rate of population that will continue in the next generation.
      mutation_rate : rate of population that will suffer mutation.
      tolerance : minimum change in cost to converge.
      n_iter : maximum number of iterations.

    Example
    -------
      >>> def f(c):
      >>>   return 300*c[0] - 100*c[1] + 200*c[2] - 500*c[3]
      >>> galg = GA(f, nvars=4, pop_size=20, tolerance = 1e-4,
      >>>           patience=5, max_iter=50)
      >>> galg.run()
      Tolerance reached!

      Number of iterations: 9
      Best chromosome: [0 1 0 1]
      Cost: -600
    """
    self.nvars = nvars
    self.gene_size = gene_size
    self.chromo_size = gene_size*nvars
    self.cost_f = cost_f
    self.pop_size = pop_size
    self.pop = np.random.choice([0,1], (pop_size, self.chromo_size))
    self.calc_cost()
    self.last_cost = np.min(self.cost, axis=0)[0]
    self.x_rate = x_rate
    self.mutation_rate = mutation_rate
    self.n_keep = round(x_rate*pop_size)
    self.select_parents()
    self.tolerance = tolerance
    self.patience = patience
    self.max_iter = max_iter

  def calc_cost(self):
    self.cost = []
    for i, chromosome in enumerate(self.pop):
      self.cost.append([self.cost_f(chromosome), i])
    self.cost.sort()

  def select_parents(self):
    parents = []
    parents_idx = []
    self.calc_cost()

    for i in range(len(self.cost)):
      parents_idx.append(self.cost[i][1])

    parents_idx = parents_idx[:self.n_keep]
    self.parents = self.pop[parents_idx]

  def mating_pool(self):
    rand_idx = np.random.shuffle(np.arange(len(self.parents)))
    self.parents = self.parents[rand_idx][0]
    self.offspring = []
    if len(self.parents)%2 == 0:
      for i in np.arange(0,len(self.parents),2):
        father = self.parents[i]
        mother = self.parents[i+1]
        cross_point = np.random.choice(np.arange(self.chromo_size-1))
        self.offspring.append([*father[:cross_point], *mother[cross_point:]])
        self.offspring.append([*mother[cross_point:], *father[:cross_point]])

    else:
      for i in np.arange(0,len(self.parents)-1,2):
        father = self.parents[i]
        mother = self.parents[i+1]
        cross_point = np.random.choice(np.arange(self.chromo_size-1))
        self.offspring.append([*father[:cross_point], *mother[cross_point:]])
        self.offspring.append([*mother[cross_point:], *father[:cross_point]])
      father = self.parents[-1]
      mother = self.parents[0]
      cross_point = np.random.choice(np.arange(self.chromo_size-1))
      self.offspring.append([*father[:cross_point], *mother[cross_point:]])
    self.offspring = np.asarray(self.offspring)

  def next_gen(self):
    self.select_parents()
    self.mating_pool()
    self.pop = np.asarray([*self.parents, *self.offspring])
    self.mutation()

  def mutation(self):
    self.calc_cost()
    self.best_chrom = self.pop[self.cost[0][1]]
    mut_idx = [int(c) for c in np.array(self.cost[1:])[:,1]]
    mut_group = self.pop[mut_idx]
    self.pop = [self.best_chrom,]
    rand_idx = np.arange(len(mut_group))
    np.random.shuffle(rand_idx)
    n_mut = round(self.mutation_rate*len(mut_group))
    rand_idx = rand_idx[:n_mut]
    for i, chromosome in enumerate(mut_group[rand_idx]):
      mut_gene = np.random.choice(np.arange(self.chromo_size))
      if chromosome[mut_gene] == 0:
        mut_group[rand_idx[i]][mut_gene] = 1
      elif chromosome[mut_gene] == 1:
        mut_group[rand_idx[i]][mut_gene] = 0
    self.pop.extend([*mut_group])
    self.pop = np.asarray(self.pop)

  def run(self):
    iterations = 0
    p = 0
    while True:
      self.calc_cost()
      self.best_chrom = self.pop[self.cost[0][1]]
      if self.tolerance:
        delta = abs(np.min(self.cost, axis=0)[0] - (np.min(self.last_cost, axis=0)))
        if delta <= self.tolerance:
          p += 1
          if p >= self.patience:
            print("Tolerance reached!\n")
            break
        else:
          p = 0
      if iterations >= self.max_iter:
        print("Maximum iterations reached!\n")
        break

      self.last_cost = np.min(self.cost, axis=0)[0]
      self.next_gen()
      self.calc_cost()
      self.best_chrom = self.pop[self.cost[0][1]]
      iterations += 1
    print("Number of iterations:", iterations)
    print("Best chromosome:", self.best_chrom)
    print("Cost:", self.cost_f(self.best_chrom))
