import numpy as np
import math
from dataclasses import dataclass 
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-0.2*x))

@dataclass
class Neurone:
    Q: np.array
    h: callable

    def forward(self, entree, poids):
        output = np.dot(poids, entree)
        return self.h(output)

@dataclass
class NN :
    data_inputs: np.array
    input_size: int
    targets: np.array
    targets_size: int
    nb_couches: int
    nb_neurones_couche: int = 1
    act_func : callable = sigmoid

    def __post_init__(self):
        self.poids = [np.random.uniform(-20, 20, size=(self.nb_neurones_couche, self.input_size))]
        self.couches = [np.array([Neurone(Q, self.act_func) for Q in self.poids[0]])]

        for k in range(self.nb_couches-1):
            self.poids.append(np.random.uniform(-20, 20, size=(self.nb_neurones_couche, self.nb_neurones_couche)))
            self.couches.append(np.array([Neurone(Q, self.act_func) for Q in self.poids[k+1]]))
        self.poids.append(np.random.uniform(-20, 20, size=(self.targets_size, self.nb_neurones_couche)))
        self.couches.append(np.array([Neurone(Q, self.act_func) for Q in self.poids[-1]]))

    def forward(self, entree, couche, poids):
        outputs = []
        for n,neurone in enumerate(couche):
            outputs.append(neurone.forward(entree, poids[n]))
        return outputs
    
    def fullforward(self, entree, poids):
        for c, couche in enumerate(self.couches):
            entree = self.forward(entree, couche, poids[c])
        return entree

    def calculer_erreur(self, entree, cible, poids):
        return np.linalg.norm(self.fullforward(entree, poids) - cible)**2


    def train(self, pop_size, nb_generations_max, erreur_toleree, nb_pm, nb_co, np_gm):
        population = self.creer_population(pop_size)
        g = 0
        erreur_min = np.inf
        while g < nb_generations_max and erreur_min > erreur_toleree:
            g+=1
            if g%100 == 0:
                print(np.sort(erreurs))
                print(g)
            
            erreurs = []
            for k in range(int(len(population))):
                total_err = 0
                for e, entree in enumerate(self.data_inputs):
                    individu = population[k]
                    if k==0 :
                        pass
                    elif k<nb_co+1:
                        individu = self.crossover(individu, population[0])
                    elif k<nb_pm+nb_co+1:
                        individu = self.petite_mutatation(individu, population[0])
                    elif k<np_gm+nb_pm+nb_co+1:
                        individu = self.grande_mutatation(individu)
                    total_err+=self.calculer_erreur(entree, self.targets[e], individu)
                erreurs.append(total_err)
            population = self.tri(population, erreurs)
            erreur_min = erreurs[0]
            
        population = self.tri(population, erreurs)
        best_poids = population[0]
        return best_poids, erreurs
    
    def creer_population(self, pop_size):
        population = []
        for k in range(pop_size):
            individu = []

            individu.append(np.random.uniform(-20, 20, size=(self.nb_neurones_couche, self.input_size)))

            for k in range(self.nb_couches-1):
                individu.append(np.random.uniform(-20, 20, size=(self.nb_neurones_couche, self.nb_neurones_couche)))
            individu.append(np.random.uniform(-20, 20, size=(self.targets_size, self.nb_neurones_couche)))
            population.append(individu)
        return population
    
    def petite_mutatation(self, individu, best_indiv):
        id_mat_poids = random.randint(0,1)
        
        for i in range(len(individu[id_mat_poids])):
            j = random.randint(0, len(individu[id_mat_poids][i])-1)
            #for j in range(len(individu[id_mat_poids][i])):
            individu[id_mat_poids][i][j] += np.random.normal(0, 1)*min(abs(20 - best_indiv[id_mat_poids][i][j]), abs(-20 + best_indiv[id_mat_poids][i][j]))/40 # Petite perturbation
        #i = random.randint(0, len(individu[id_mat_poids])-1)
        #j = random.randint(0, len(individu[id_mat_poids][i])-1)
        
        #individu[id_mat_poids][i][j] += np.random.normal(0, 1)*min(abs(20 - best_indiv[id_mat_poids][i][j]), abs(-20 + best_indiv[id_mat_poids][i][j]))/40 # Petite perturbation

        return individu
    
    def grande_mutatation(self, individu):
        individu[0] = np.random.uniform(-20, 20, size=(self.nb_neurones_couche, self.input_size))
        individu[1] = np.random.uniform(-20, 20, size=(self.targets_size, self.nb_neurones_couche))
        """for mat_poids in individu:
            for i in range(len(mat_poids)):
                for j in range(len(mat_poids[i])):
                    mat_poids[i][j] = np.random.uniform(-20, 20) # Mutation aléatoire sur une large échelle"""
        return individu
    
    def crossover(self, parent1, parent2):
        child = []
        for i in range(len(parent1)):
            if i % 2 == 0:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    def tri(self, population, erreurs):
        sorted_indices = np.argsort(erreurs)
        population_new = [population[i] for i in sorted_indices] 
        return population_new
    
def main():
    
    phi = 0.21
    ksi = 0.4

    inputs = np.array([[(math.sin(phi*j/15*i*2*3.141592+ksi)+1)/2 for j in range(15)] for i in range(6)])
    targets = np.eye(6)

    pop_size = 20
    nb_generations_max = 300000
    erreur_max = 1
    nb_pm = 5
    nb_co = 10
    np_gm = 5

    nn = NN(data_inputs=inputs, input_size=15, targets=targets, targets_size=6, nb_couches=1, nb_neurones_couche=23)
    poids, liste_erreurs = nn.train(pop_size, nb_generations_max, erreur_max, nb_pm, nb_co, np_gm)

    """for entree in inputs:
        print(poids)
        print(nn.fullforward(entree, poids))"""
    
    iterations = range(1, len(liste_erreurs) + 1)
    plt.plot(iterations, liste_erreurs, marker='o', linestyle='-')
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Erreur')
    plt.title('Diminution de l\'erreur lors de l\'entraînement')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
