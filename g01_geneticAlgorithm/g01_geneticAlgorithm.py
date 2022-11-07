import random as rnd
import math


class Chromosome:

    def __init__(self, f1, f2, f3, f4, f5):
        # all features as bit strings
        self.f1 = f1  # [0-45] propeller blade angle
        self.f2 = f2  # [2,3,4,5] number of blades
        self.f3 = f3  # [0.55-0.75] air/fuel ratio (x100)
        self.f4 = f4  # [1000-2000] propeller diameter
        self.f5 = f5  # [0.5-5.0] idle valve position (x10)
        self.fitness = float('inf')

        # following code is important to check in each generation that a new offspring has valid values for the features
        self.f1 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(f1), 0, 45))
        self.f2 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(f2), 2, 5))
        self.f3 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(f3), 55, 75))
        self.f4 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(f4), 1000, 2000))
        self.f5 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(f5), 5, 50))

    def map_int_to_bin(self, val):
        return format(val, 'b')

    def map_bin_to_int(self, val):
        return int(val, 2)

    def map_to_f3(self, val):
        return self.map_bin_to_int(val) / 100

    def map_to_f5(self, val):
        return self.map_bin_to_int(val) / 10

    def set_fitness(self, new_fitness):
        self.fitness = new_fitness

    def get_fitness(self):
        return self.fitness

    def crossover(self, other, fun):
        return (Chromosome(fun(self.f1, other.f1),
                           fun(self.f2, other.f2),
                           fun(self.f3, other.f3),
                           fun(self.f4, other.f4),
                           fun(self.f5, other.f5)))

    def arithmetic_mean(self, val1, val2):
        return self.map_int_to_bin(round((self.map_bin_to_int(val1) + self.map_bin_to_int(val2)) / 2))

    def single_point(self, val1, val2):
        val1, val2 = self.fill_bitstring(val1, val2)
        crossover_point = rnd.randint(0, len(val1))
        child = val1[:crossover_point] + val2[crossover_point:]
        return child

    def multi_point(self, val1, val2):
        val1, val2 = self.fill_bitstring(val1, val2)
        parents = [val1, val2]
        child = ''
        for bit in range(len(val1)):
            child += parents[rnd.randint(0, 1)][bit]
        return child

    def fill_bitstring(self, val1, val2):
        while len(val1) < len(val2):
            val1 = '0' + val1
        while len(val2) < len(val1):
            val2 = '0' + val2
        return val1, val2

    def mutate_random_offset(self):
        chance = rnd.random()
        if chance <= mutation_rate:
            f1 = self.map_bin_to_int(self.f1)
            f1 += rnd.randint(-10, 10)
            f1 = self.check_feature_min_max(f1, 0, 45)
            self.f1 = self.map_int_to_bin(f1)

            f2 = self.map_bin_to_int(self.f2)
            f2 += rnd.randint(-1, 1)
            f2 = self.check_feature_min_max(f2, 2, 5)
            self.f2 = self.map_int_to_bin(f2)

            f3 = self.map_bin_to_int(self.f3)
            f3 += rnd.randint(-3, 3)
            f3 = self.check_feature_min_max(f3, 55, 75)
            self.f3 = self.map_int_to_bin(f3)

            f4 = self.map_bin_to_int(self.f4)
            f4 += rnd.randint(-100, 100)
            f4 = self.check_feature_min_max(f4, 1000, 2000)
            self.f4 = self.map_int_to_bin(f4)

            f5 = self.map_bin_to_int(self.f5)
            f5 += rnd.randint(-5, 5)
            f5 = self.check_feature_min_max(f5, 5, 50)
            self.f5 = self.map_int_to_bin(f5)

    def check_feature_min_max(self, feature, min_value, max_value):
        if feature > max_value:
            return max_value
        elif feature < min_value:
            return min_value
        else:
            return feature

    def bitflip(self, val1):
        index = rnd.randint(0, len(val1) - 1)
        if val1[index] == '0':
            return val1[:index] + '1' + val1[index + 1:]
        else:
            return val1[:index] + '0' + val1[index + 1:]

    def mutate_bitflip(self):
        chance = rnd.random()
        if chance <= mutation_rate:
            self.f1 = self.bitflip(self.f1)
            self.f1 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(self.f1), 0, 45))
            self.f2 = self.bitflip(self.f2)
            self.f2 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(self.f1), 2, 5))
            self.f3 = self.bitflip(self.f3)
            self.f3 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(self.f1), 55, 75))
            self.f4 = self.bitflip(self.f4)
            self.f4 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(self.f1), 1000, 2000))
            self.f5 = self.bitflip(self.f5)
            self.f5 = self.map_int_to_bin(self.check_feature_min_max(self.map_bin_to_int(self.f1), 5, 50))

    def eval(self):
        return (self.map_bin_to_int(self.f1) ** self.map_bin_to_int(self.f2) + math.log(self.map_to_f3(self.f3))) \
               / (self.map_bin_to_int(self.f4) + self.map_to_f5(self.f5) ** 3)

    def __str__(self):
        return ('   Thrust: {} --blade angle: {}° --number of blades: {} --air/fuel ratio: {} '
                '--propeller diameter: {}mm --idle valve position: {}') \
            .format(self.eval(), self.map_bin_to_int(self.f1), self.map_bin_to_int(self.f2), self.map_to_f3(self.f3),
                    self.map_bin_to_int(self.f4), self.map_to_f5(self.f5))


def elitist_selection(population):
    for j in range(1, 21):
        new_offspring = population[j - 1].crossover(population[j + 1], population[j - 1].multi_point)
        new_offspring.mutate_bitflip()
        population[len(population) - j] = new_offspring
    return population


def roulette_wheel_selection(population):
    new_population = []
    for i in range(len(population)):
        chance = rnd.randint(0, 500)
        if population[i].get_fitness() > chance:
            random_parents = rnd.sample(range(0, len(population)), 2)
            new_offspring = (population[random_parents[0]].crossover(population[random_parents[1]],
                                                                     population[random_parents[0]].multi_point))
            new_offspring.mutate_bitflip()
            new_population.append(new_offspring)
        else:
            new_population.append(population[i])
    return new_population


def run_model(pop):
    pop_sum = 0
    for p in pop:
        f = abs(p.eval() - thrust_value)
        p.set_fitness(f)
        pop_sum += f

    av_fit = pop_sum / len(pop)
    pop.sort(key=lambda x: x.get_fitness())
    if selection_nr == '1':
        pop = elitist_selection(pop)
    elif selection_nr == '2':
        pop = roulette_wheel_selection(pop)
    print('Average population fitness:' + str(av_fit) + '\n--- Best individual:' + str(pop[0]))

    return pop, av_fit


population = []
print('---- WELCOME ---- \n## Enjoy the Genetic Algorithm, designed by "The Amazing Flying Chicken" ##')
pop_size = int(input('Please enter a population size: '))
mutation_rate = float(input('Please enter a mutation rate: '))
thrust_value = int(input('Please enter the optimal thrust value: '))
all_selections = ['Elitist', 'Roulette wheel']
selection_nr = input('What selection scheme to use? 1=Elitist | 2=Roulette wheel ')
selection = all_selections[int(selection_nr) - 1]
termination_criteria = input('When should the program stop? 1=Number of generations | 2=Average fitness ')

for i in range(pop_size):
    population.append(Chromosome(format(rnd.randint(0, 45), 'b'),
                                 format(rnd.randint(2, 5), 'b'),
                                 format(rnd.randint(55, 75), 'b'),
                                 format(rnd.randint(1000, 2000), 'b'),
                                 format(rnd.randint(5, 50), 'b')))

average_fitness = float('inf')

if termination_criteria == '1':
    generations = int(input('Please enter the number of generations: '))
    print('----------------------------------\nThe model is being trained using the following inputs:')
    print(('--Population size: {}    --Mutation rate: {}%    --Optimal thrust: {}    --Selection scheme: {}'
           '    --Termination after {} generations')
          .format(pop_size, mutation_rate * 100, thrust_value, selection, generations))
    input('----------------------------------\nPress enter to continue')
    for gen in range(generations):
        print('generation', gen + 1)
        population, average_fitness = run_model(population)

elif termination_criteria == '2':
    desired_fitness = int(input('Please enter the desired average fitness: '))
    print('----------------------------------\nThe model is being trained using the following inputs:')
    print(('--Population size: {}    --Mutation rate: {}%    --Optimal thrust: {}    --Selection scheme: {}'
           '    --Termination after reaching an average fitness of {}')
          .format(pop_size, mutation_rate * 100, thrust_value, selection, desired_fitness))
    input('----------------------------------\nPress enter to continue')
    while average_fitness > desired_fitness:
        population, average_fitness = run_model(population)
