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
            for s in range(4, 22)
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

    def mejor_suma_crupier(self):
        """
        Calcula la suma óptima del crupier considerando que los ases
        pueden valer 1 u 11 (se usan como 11 mientras no excedan 21).
        """
        valores = [c for c, v in self.mazo_crupier]
        suma = sum(valores)
        num_ases = valores.count(1)
        while suma + 10 <= 21 and num_ases > 0:
            suma += 10
            num_ases -= 1
        return suma

    def turno_crupier(self):
        """
        Pide carta mientras su suma sea menor a 17 y retorna la suma final del crupier.
        """
        while self.mejor_suma_crupier() < 17:
            self.mazo_crupier = self.agregar_carta(self.mazo_crupier, self.reparte_carta())
        return self.mejor_suma_crupier()

    def recompensa(self, s, a, s_):
        """
        La recompensa en cada estado no terminal es 0.

        Para estados terminales:
        +1: El jugador gana.
        0: Empate (Push).
        -1: El jugador pierde o se pasa (Bust).
        +1.5: Victoria por Blackjack natural (21 con las dos primeras cartas).
        """
        if not self.es_terminal(s_):
            return 0

        player_sum = s_[0]

        if player_sum > 21:
            return -1

        dealer_sum = s_[1]

        if self.natural:
            return 1.5

        if dealer_sum > 21 or player_sum > dealer_sum:
            return 1
        elif player_sum == dealer_sum:
            return 0
        else:
            return -1

    def transicion(self, s, a):
        """
        Calcula la transición dado el estado actual s y la acción a.
        Donde:
        Si pide (a=1): Reparte una carta al jugador y actualiza la suma, si supera 21, retorna estado terminal (bust).
        Si se planta (a=0): Ejecuta el turno del crupier y retorna estado terminal con la suma final del crupier.
        """
        suma, carta_crupier, as_usable = s[0], s[1], s[2]

        if a == 1:
            self.mazo_player = self.agregar_carta(self.mazo_player, self.reparte_carta())
            nueva_as_usable = self.as_usable()
            nueva_suma = self.sumar_mazo(self.mazo_player)

            if nueva_suma > 21:
                return (nueva_suma, carta_crupier, nueva_as_usable, True)
            return (nueva_suma, carta_crupier, nueva_as_usable)

        else:
            dealer_final = self.turno_crupier()
            return (suma, dealer_final, as_usable, True)

    def es_terminal(self, s):
        """
        Un estado se considera terminal cuando la tupla contiene 4 elementos,
        donde el cuarto valor es True, esto sucede cuando el jugador se pasa
        de 21 (bust) o decide plantarse.
        """
        return len(s) == 4


if __name__ == "__main__":

    blackjack = BlackJack(gama=1)

    # de cada algoritmo
    Q_sarsa = SARSA(blackjack, alfa=0.1, epsilon=0.05, n_ep=500000, n_iter=100)
    Q_ql = Q_learning(blackjack, alfa=0.1, epsilon=0.05, n_ep=500000, n_iter=100)

    # Encuentra las políticas óptimas para cada algoritmo
    pi_s = PoliticaGreedy(Q_sarsa)
    pi_q = PoliticaGreedy(Q_ql)

    # Imprime las políticas óptimas para cada estado no terminal
    print("Estado".center(22) + '|' + "SARSA".center(10) + '|' + "Q-learning".center(10))
    print("-" * 22 + '|' + "-" * 10 + '|' + "-" * 10)
    for s in blackjack.estados:
        if not blackjack.es_terminal(s):
            print(str(s).center(22) + '|'
                  + str(pi_s(s)).center(10) + '|'
                  + str(pi_q(s)).center(10))
    print("-" * 22 + '|' + "-" * 10 + '|' + "-" * 10)

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