import numpy as np

class GA:
  def __init__(self, cost_f, gene_sizes, pop_size=8, mutation_rate=0.2, 
               tolerance=None, patience=5, max_iter=1e4, verbose=0):
    """
    Genetic Algorithm for binary variables.

    Parameters
    -----------
      cost_f : cost function, must have a list as parameter and return the cost.
      gene_sizes : list of variables sizes.
      pop_size : population size. number of chromosomes in each generation.
      mutation_rate : percentage of bits that will suffer mutation.
      tolerance : minimum change in cost to converge.
      n_iter : maximum number of iterations.

    Example
    -------
      >>> def f(c):
      >>>   return 300*c[0] - 100*c[1] + 200*c[2] - 500*c[3]
      >>> galg = GA(f, gene_sizes=4, pop_size=20, tolerance = 1e-4,
      >>>           patience=5, max_iter=50)
      >>> galg.run()
      Tolerance reached!

      Number of iterations: 9
      Best chromosome: [0 1 0 1]
      Cost: -600
    """
    np.random.seed(None)
    self.gene_sizes = gene_sizes
    self.chromo_size = gene_sizes if str(type(gene_sizes))=="<class 'int'>" else sum(gene_sizes)
    self.cost_f = cost_f
    self.pop_size = pop_size
    self.pop = np.random.choice([0,1], (pop_size, self.chromo_size))
    self.calc_cost()
    self.last_cost = np.min(self.cost, axis=0)[0]
    self.x_rate = 0.5
    self.mutation_rate = mutation_rate
    self.n_mutation = round(self.mutation_rate*(self.pop_size-1)*self.chromo_size)
    self.n_keep = round(self.x_rate*pop_size)
    self.select_parents()
    self.tolerance = tolerance
    self.patience = patience
    self.max_iter = max_iter
    self.verbose = verbose
    
  def calc_cost(self):
    self.cost = []
    for i, chromosome in enumerate(self.pop):
      self.cost.append([self.cost_f(chromosome), i])
    self.cost.sort()

  def select_parents(self):
    parents = []
    parents_idx = []
    #self.calc_cost()

    for i in range(len(self.cost)):
      parents_idx.append(self.cost[i][1])

    parents_idx = parents_idx[:self.n_keep]
    self.parents = self.pop[parents_idx]

  def mating_pool(self):
    np.random.seed(None)
    rand_idx = np.random.shuffle(np.arange(len(self.parents)))
    self.parents = self.parents[rand_idx][0]
    self.offspring = []
    
    if len(self.parents)%2 == 0:
      for i in np.arange(0,len(self.parents),2):
        father = self.parents[i]
        mother = self.parents[i+1]
        cross_point = np.random.choice(np.arange(self.chromo_size-1))
        self.offspring.append([*father[:cross_point], *mother[cross_point:]])
        self.offspring.append([*mother[:cross_point], *father[cross_point:]])

    else:
      for i in np.arange(0,len(self.parents)-1,2):
        father = self.parents[i]
        mother = self.parents[i+1]
        cross_point = np.random.choice(np.arange(self.chromo_size-1))
        self.offspring.append([*father[:cross_point], *mother[cross_point:]])
        self.offspring.append([*mother[:cross_point], *father[cross_point:]])
      father = self.parents[-1]
      mother = self.parents[0]
      cross_point = np.random.choice(np.arange(self.chromo_size-1))
      self.offspring.append([*father[:cross_point], *mother[cross_point:]])
    self.offspring = np.asarray(self.offspring)

  def next_gen(self):
    self.select_parents()
    self.mating_pool()
    self.best_chrom = self.pop[self.cost[0][1]]
    self.pop = np.asarray([*self.parents, *self.offspring])
    self.mutation()

  def mutation(self):
    np.random.seed(None)
    #self.calc_cost()
    #self.best_chrom = self.pop[self.cost[0][1]]
    mut_idx = [int(c) for c in np.array(self.cost[1:])[:,1]]
    mut_group = self.pop[mut_idx]
    self.pop = [self.best_chrom,]
    rand_idx = np.random.choice(np.arange(len(mut_group)), self.n_mutation)
    #np.random.shuffle(rand_idx)
    #n_mut = int(self.mutation_rate*len(mut_group))
    #rand_idx = rand_idx[:n_mut]
    for i, chromosome in enumerate(mut_group[rand_idx]):
      #for k in range(np.random.choice(np.arange(1,self.max_mutation))):
      chromosome
      mut_gene = np.random.choice(np.arange(self.chromo_size))
      if chromosome[mut_gene] == 0:
        mut_group[rand_idx[i]][mut_gene] = 1
      elif chromosome[mut_gene] == 1:
        mut_group[rand_idx[i]][mut_gene] = 0
    self.pop.extend([*mut_group])
    self.pop = np.asarray(self.pop)
    
  def print_partial(self):
    if self.verbose == 1:
      print("Iteration {} | Best Fitness: {}".format(self.iterations, self.cost[0][0]))
    elif self.verbose == 2:
      print("Iteration {} | Best Fitness: {}".format(self.iterations, self.cost[0][0]))
      print("Population:")
      for i, c in enumerate(self.pop):
        print("{} -> {:.2f}".format(self.pop[self.cost[i][1]], self.cost[i][0]))    

  def run(self):
    self.iterations = 0
    self.print_partial()
    p = 0
    while True:
      self.last_cost = np.min(self.cost, axis=0)[0]
      self.next_gen()
      self.calc_cost()
      self.best_chrom = self.pop[self.cost[0][1]]
      if self.tolerance:
        delta = abs(np.min(self.cost, axis=0)[0] - (np.min(self.last_cost, axis=0)))
        if delta <= self.tolerance:
          p += 1
          if p > self.patience:
            self.conv = 'tolerance'
            self.next_gen()
            self.calc_cost()
            self.best_chrom = self.pop[self.cost[0][1]]
            break
        else:
          p = 0
      if self.iterations >= self.max_iter:
        self.conv = 'maximum number of iterations'
        self.next_gen()
        self.calc_cost()
        self.best_chrom = self.pop[self.cost[0][1]]
        break

      self.iterations += 1
      self.print_partial()
    
    if not self.verbose == 0:
      print("Finished by", self.conv)
      print("Number of iterations:", self.iterations)
      print("Best chromosome:", self.best_chrom)
      print("Cost:", self.cost[0][0])
    
    return self.cost[0][0], self.best_chrom, self.conv