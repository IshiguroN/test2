# %%
import matplotlib
matplotlib.use('nbagg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import copy
import enum
from matplotlib.collections import LineCollection
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

# %%
class PlotObject(metaclass = ABCMeta):
    @abstractmethod
    def draw(self):
        return NotImplementedError

# %%
class Canvas(PlotObject):
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []  
        self.debug = debug
        self.time_span = time_span  
        self.time_interval = time_interval 
        
    def append(self,obj):  
        self.objects.append(obj)
    
    def draw(self): 
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')             
        ax.set_xlim(-10,10)                  
        ax.set_ylim(-10,10) 
        ax.set_xlabel("X",fontsize=10)                 
        ax.set_ylabel("Y",fontsize=10)                 
        
        elems = []
        
        if self.debug:        
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            
            self.ani.save("sample.gif", writer="imagemagic")
            plt.close()
        
    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "[%dst generation]" % (i)
        elems.append(ax.text(-9.0, 11.0, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)

# %%
class City(PlotObject):
    def __init__(self, x, y):
        self.pos = np.array([x,y]).T
        self.id = None
    
    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="cities", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], str(self.id), fontsize=10))

# %%
class Map(PlotObject):
    def __init__(self, num_of_cities):
        self.cities = []
        self.num_of_cities = num_of_cities
    
    def append_city(self, city):
        city.id = len(self.cities)
        self.cities.append(city)
    
    def generate_map(self, seed=100):
        np.random.seed(seed=seed)
        for _ in range(self.num_of_cities):
            self.append_city(City(np.random.rand()*20 - 10, np.random.rand()*20 - 10))

    def draw(self, ax, elems):
        for c in self.cities : c.draw(ax, elems)

# %%
class Agent(PlotObject):
    def __init__(self, map, start_id = 0):
        self.map = map
        self.start_point_id = start_id
        self.start_city_pos = [city.pos for city in self.map.cities if city.id == self.start_point_id][0]
        self.root_list = np.array([e.id for e in self.map.cities if e.id != self.start_point_id])
        self.distance = 0.0

    def optimize_root(self):
        np.random.shuffle(self.root_list)
        print(self.root_list)
    
    def get_norm(self,x,y):
        return np.linalg.norm(x-y)
    
    def calculate_distance(self, root_list):
        distance = 0.0
        distance += self.get_norm(self.start_city_pos, self.map.cities[root_list[0]].pos)
        for i in range(0, len(root_list)-1):
            distance += self.get_norm(self.map.cities[root_list[i]].pos, self.map.cities[root_list[i+1]].pos)
        distance += self.get_norm(self.map.cities[root_list[i+1]].pos, self.start_city_pos)
        return distance
    
    def draw(self, ax, elems):

        ## draw the calculated root
        lines = [] 
        lines.append([self.start_city_pos, self.map.cities[self.root_list[0]].pos])
        for i in range(0, len(self.root_list)-1):
            lines.append([self.map.cities[self.root_list[i]].pos, self.map.cities[self.root_list[i+1]].pos])
        lines.append([self.map.cities[self.root_list[i+1]].pos, self.start_city_pos])

        lc = LineCollection(lines)
        elems.append(ax.add_collection(lc))
        elems.append(ax.text(-1.0, 11.0, "the minimize distance:%.2f" % (self.distance), fontsize=10))
    
    def one_step(self, time_interval):
        self.optimize_root()
        self.distance = self.calculate_distance(self.root_list)


# %%
class Generation:
    def __init__(self):
        self.individuals = []
        self.weights = []
    
    def best_individual(self):
        weights = np.array(self.weights)
        max_index = np.argmax(weights)
        return self.individuals[max_index]
    
    def set_member(self, individuals, weights):
        self.individuals = individuals
        self.weights = weights
    
    def append_individuals(self, individual):
        self.individuals.append(individual)
    
    def get_length_of_individuals(self):
        return len(self.individuals)

    def get_weights(self):
        return self.weights
    
    def get_individuals_as_ndarray(self):
        return np.array(self.individuals)

# %%
@enum.unique
class Operation(enum.Enum):
    cross = "cross"
    mutation = "mutation"
    reproduction = "reproduction"

# %%
import sys

class GaAgent(Agent):
    def __init__(self, map, start_id = 0, num_of_individuals=10, num_of_generations=20, cross_prob=0.8, mutate_prob=0.05):
        super().__init__(map, start_id=start_id)
        self.num_of_individuals = num_of_individuals
        self.num_of_generations = num_of_generations
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.buffa = None

    def _create_generation(self):
        ge = Generation()
        individuals = []
        weights = []
        for _ in range(self.num_of_individuals):
            individual = np.random.choice(self.root_list, size=len(self.root_list),replace=False)
            individuals.append(individual)
            weights.append(self.calculate_distance(individual))
        _weights = [e/sum(weights) for e in weights]
        ge.set_member(individuals, _weights)
        return ge

    def _crossing(self, couple):
        one, other = couple
        for k in range(len(one)):
            jadge = np.random.rand()
            if jadge < 0.5:
                tmp = other[k]
                other[k] = one[k]
                one[k] = tmp
        return (one, other)
    
    def _mutation(self, individual):
        one = individual
        former, latter = np.random.choice([i for i in range(len(one))], 2, replace=False)
        one[former], one[latter] = one[latter], one[former]
        return one

    def _reproduction(self, individual):
        return copy.deepcopy(individual)

    # @override
    def optimize_root(self):
        if self.buffa == None:
            now_ge = self._create_generation()
        else:
            now_ge = self.buffa
        now_individuals = now_ge.individuals
        next_ge = Generation()
        i = 0
        while i < self.num_of_individuals:
            operation = np.random.choice([Operation.cross, Operation.mutation, Operation.reproduction], 1, p=[self.cross_prob, self.mutate_prob, 1-self.cross_prob-self.mutate_prob])[0]
            print(operation)
            print(now_individuals)
            if operation == Operation.cross:
                one, other = np.random.choice([1 for i in range(len(now_ge.get_weights()))], 2, p=now_ge.get_weights())
                # (one, other) = now_individuals[np.random.multinomial(2, now_ge.get_weights())==1]

                print("passed")
                atuple = self._crossing((one, other))
                print(atuple[0], atuple[1])
                next_ge.append_individuals(atuple[0])
                next_ge.append_individuals(atuple[1])

            elif operation == Operation.mutation:
                one = now_individuals[np.random.multinomial(1, now_ge.get_weights())==1][0]
                print(one)
                next_ge.append_individuals(self._mutation(one))

            elif operation == Operation.reproduction:
                one = now_individuals[np.random.multinomial(1, now_ge.get_weights())==1][0]
                next_ge.append_individuals(self._reproduction(one))
            else:
                one = now_individuals[np.random.multinomial(1, now_ge.get_weights())==1][0]
                next_ge.append_individuals(self._reproduction(one))
            print()
            print("debug: ")
            for e in next_ge.individuals:
                print(e)
                len(e)
                
            i+=1
        
        print("debug")
        self.root_list = now_ge.best_individual()
        self.buffa = next_ge


# %%
def main():
    num_of_cities = 10
    generation_steps = 1

    canvas = Canvas(generation_steps, 3.0)
    m = Map(num_of_cities)
    m.generate_map()
    
    canvas.append(m)
    agent = GaAgent(m)

    canvas.append(agent)
    
    canvas.draw()

if __name__ == "__main__":
    main()

# %%
# def my_callback(i, elems, ax):
#     while elems: elems.pop().remove()
#     print(i)
#     time_str = "hoge %.2f" % (i*0.1)
#     elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))


# def main():
#     fig = plt.figure(figsize=(4,4))
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')             
#     ax.set_xlim(-5,5)                  
#     ax.set_ylim(-5,5) 
#     ax.set_xlabel("X",fontsize=10)                 
#     ax.set_ylabel("Y",fontsize=10)                 
    
#     elems = []
    
#     ani = anm.FuncAnimation(fig, my_callback, fargs=(elems, ax), frames=int(100), interval=int(100), repeat=False)
#     ani.save("./sample.gif", writer="imagemagick")
#     plt.show()
    

# %% [markdown]
# 


