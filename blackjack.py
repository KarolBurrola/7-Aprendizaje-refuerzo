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
        Inicializando
        """
        self.estados = [
            (s, c, a)
            for s in range(12, 22)
            for c in range(1, 11)
            for a in [True, False]
        ]
        self.mazo_player = []
        self.mazo_crupier = []
        self.natural = False
        super().__init__(self.estados, gama)

    def reparte_carta(self):
        """
        Función auxiliar que devuelve un valor aleatorio entre 1 y 10 (con probabilidad 4/13
        para el valor 10)
        """
        carta = randint(1, 13)
        return 10 if carta > 10 else carta

    def estado_inicial(self):
        """
        Reinicia el mazo y reparte dos cartas al jugador y dos al crupier (una oculta).
        Calcula el estado inicial, verifica si hay Blackjack natural inmediato (suma
        == 21 con las primeras dos cartas iniciales).
        """
        self.mazo_player = []
        self.mazo_crupier = []
        self.natural = False

        self.mazo_player = self.agregar_carta(self.mazo_player, self.reparte_carta())
        self.mazo_player = self.agregar_carta(self.mazo_player, self.reparte_carta())
        self.mazo_crupier = self.agregar_carta(self.mazo_crupier, self.reparte_carta())
        self.mazo_crupier = self.agregar_carta(self.mazo_crupier, self.reparte_carta(), visible=False)

        as_u = self.as_usable()
        suma_jugador = self.sumar_mazo(self.mazo_player)

        if suma_jugador == 21:
            self.natural = True

        return (suma_jugador, self.obtener_carta_crupier(), as_u)

    def as_usable(self):
        """
        Contabiliza el As como 11 si no excede 21. Si la suma excede 21 y se tiene un As de valor 11,
        lo transforma automáticamente a valor 1 y cambia el estado a As Usable = False.
        Retorna True si hay un as usable (contado como 11), False en caso contrario.
        """
        index_ases = [i for i, (c, v) in enumerate(self.mazo_player) if c == 1 or c == 11]
        suma_jugador = self.sumar_mazo(self.mazo_player)

        if suma_jugador > 21:
            for i in index_ases:
                c, v = self.mazo_player[i]
                if c == 11:
                    self.mazo_player[i] = (1, v)
                    return False
        else:
            for i in index_ases:
                c, v = self.mazo_player[i]
                if c == 1 and suma_jugador + 10 <= 21:
                    self.mazo_player[i] = (11, v)
                    return True

        return False

    def agregar_carta(self, mazo, c1, visible=True):
        mazo.append((c1, visible))
        return mazo

    def sumar_mazo(self, mazo):
        return sum([c for c, v in mazo])

    def obtener_carta_crupier(self):
        c, v = self.mazo_crupier[0]
        return c

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

    blackjack = BlackJack(gama=1, ...)  # TODO: agregar los parámetros necesarios para el blackjack

    # TODO: definir los parámetros de SARSA y Q-learning, luego crear las instancias
    # de cada algoritmo
    Q_sarsa = SARSA(blackjack, alfa=..., epsilon=..., n_ep=..., n_iter=...)
    Q_learning = Q_learning(blackjack, alfa=..., epsilon=..., n_ep=..., n_iter=...)

    # Encuentra las políticas óptimas para cada algoritmo
    pi_s = PoliticaGreedy(Q_sarsa)
    pi_q = PoliticaGreedy(Q_learning)

    # Imprime las políticas óptimas para cada estado no terminal
    print("Estado".center(10) + '|' + "SARSA".center(10) + '|' + "Q-learning".center(10))
    print("-" * 10 + '|' + "-" * 10 + '|' + "-" * 10)
    for s in blackjack.estados:
        if not blackjack.es_terminal(s):
            print(str(s).center(10) + '|'
                  + str(pi_s(s)).center(10) + '|'
                  + str(pi_q(s)).center(10))
    print("-" * 10 + '|' + "-" * 10 + '|' + "-" * 10)

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