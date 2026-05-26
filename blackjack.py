"""
El problema del blackjack simplificado como un problema de aprendizaje por refuerzo

"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import random, randint

class BlackJack(MDPsim):
    """
    Clase que representa un MDP para el problema del Blackjack.
    """
    def __init__(self, gama):
        """
        Inicializando variables
        """
        self.estados = []
        for suma in range(12, 22):
            for carta_crupier in range(1, 11):
                for as_usable in [True, False]:
                    estado = (suma, carta_crupier, as_usable)
                    self.estados.append(estado)

        self.gama = gama
        self.carta_c_oculta = None
        
    def estado_inicial(self):
        # TODO: implementar el estado inicial del blackjack
        raise NotImplementedError("Implementa el estado inicial del blackjack")
    
    def acciones_legales(self, s):
        plantarse = 0
        pedir = 1
        return [plantarse, pedir]
    
    def recompensa(self, s, a, s_):
        # TODO: implementar la recompensa del blackjack
        raise NotImplementedError("Implementa la recompensa del blackjack")
    
    def transicion(self, s, a):
        # TODO: implementar la transición del blackjack
        raise NotImplementedError("Implementa la transición del blackjack")
    
    def es_terminal(self, s):
        # TODO: implementar la condición de estado terminal del blackjack
        raise NotImplementedError("Implementa la condición de estado terminal del blackjack")


if __name__ == "__main__":

    blackjack = BlackJack(gama=1,...) # TODO: agregar los parámetros necesarios para el blackjack   

    # TODO: definir los parámetros de SARSA y Q-learning, luego crear las instancias 
    # de cada algoritmo
    Q_sarsa = SARSA( blackjack, alfa=..., epsilon=..., n_ep=..., n_iter=...)
    Q_learning = Q_learning( blackjack, alfa=..., epsilon=..., n_ep=..., n_iter=...)

    # Encuentra las políticas óptimas para cada algoritmo
    pi_s = PoliticaGreedy(Q_sarsa)
    pi_q = PoliticaGreedy(Q_learning)

    # Imprime las políticas óptimas para cada estado no terminal
    print("Estado".center(10) + '|' +  "SARSA".center(10) + '|' + "Q-learning".center(10))
    print("-"*10 + '|' + "-"*10 + '|' + "-"*10)
    for s in blackjack.estados:
        if not blackjack.es_terminal(s):
            print(str(s).center(10) + '|' 
                  + str(pi_s(s)).center(10) + '|' 
                  + str(pi_q(s)).center(10))
    print("-"*10 + '|' + "-"*10 + '|' + "-"*10)


"""
****************************************************************************************
Responde las siguientes preguntas:

1. ¿Cuáles son los estados, acciones, recompensas y transiciones en el problema del 
    blackjack?  

2. ¿Cómo se pueden representar los estados del blackjack de manera eficiente para el 
    aprendizaje por refuerzo?

3. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?

4. ¿Cómo afecta el valor de alfa en la convergencia de los algoritmos SARSA y Q-learning?

5. ¿Cuál de los dos algoritmos, SARSA o Q-learning, consideras que es más adecuado para 
   el problema del blackjack y por qué?

6. ¿Se puede explicar con cierta lógica del juego la política óptima encontrada por cada 
   algoritmo? ¿Qué acciones se toman en cada estado y por qué?
****************************************************************************************
"""
