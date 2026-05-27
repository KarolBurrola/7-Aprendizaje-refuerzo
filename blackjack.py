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

1. ¿Cuáles son los estados, acciones, recompensas y transiciones en el problema del blackjack?  
Estados: Una tupla de 3 elementos compuesta por: suma_jugador, carta_visible_crupier, as_usable.
Donde la suma_jugador va de 4 a 21 en la implementación, aunque los estados relevantes para decisiones no triviales 
son de 12 a 21 (con suma menor a 12 siempre conviene pedir sin importar nada más), carta_visible_crupier va de 1 al 10
y as_usable es (True/False). La cardinalidad teórica del problema es 10 x 10 x 2 = 200 estados (sumas 12–21), pero en 
la implementación se amplió a 18 x 10 x 2 = 360 para cubrir sumas desde 4 y evitar errores durante el entrenamiento.

Acciones: Solo hay dos posibles, plantarse que la tomamos como un equivalente a 0, y pedir que la tomamos como 1

Recompensas:
La recompensa en cada estado no terminal es 0.
Para estados terminales:
+1: El jugador gana.
0: Empate (Push).
-1: El jugador pierde o se pasa (Bust).
+1.5: Victoria por Blackjack natural (21 con las dos primeras cartas).

Transiciones: 
Si pide (a=1): Reparte una carta al jugador y actualiza la suma, si supera 21, retorna estado terminal (bust).
Si se planta (a=0): Ejecuta el turno del crupier y retorna estado terminal con la suma final del crupier.

2. ¿Cómo se pueden representar los estados del blackjack de manera eficiente para el aprendizaje por refuerzo?
Una tupla de 3 elementos compuesta por: suma_jugador, carta_visible_crupier, as_usable. Esta representación resulta 
eficiente ya que almacena únicamente la información indispensable para decidir la siguiente acción, sin necesidad de 
conservar el historial completo de cartas obtenidas. Para el jugador, basta con conocer la suma actual de su mano y 
si dispone de un as utilizable.


3. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
Epsilon controla el equilibrio entre la exploración y la explotación dentro del proceso de aprendizaje.
La exploración consiste en que el agente pruebe acciones nuevas o aleatorias para conocer más su entorno
y descubrir posibles mejores estrategias. En cambio, la explotación ocurre cuando el agente utiliza el 
conocimiento adquirido para seleccionar las acciones que considera más convenientes y así maximizar su recompensa.

Epsilon alto: El agente realiza una mayor exploración, eligiendo acciones aleatorias más seguido, donde esto le 
permite conocer mejor el espacio de estados, aunque la convergencia de la política suele ser más lenta y menos 
estable, ya que continúa explorando incluso después de haber encontrado buenas opciones.

Epsilon bajo: El agente se enfoca en aprovechar el conocimiento que ya posee, eligiendo en la mayoría de los casos 
la acción que considera mejor, esto permite que la política alcance la convergencia con mayor rapidez, sin embargo, 
si la exploración inicial fue insuficiente, puede terminar aprendiendo una estrategia que no sea realmente la óptima.

4. ¿Cómo afecta el valor de alfa en la convergencia de los algoritmos SARSA y Q-learning?
El parámetro alfa representa la tasa de aprendizaje del agente, es decir, qué tanto influyen las nuevas experiencias 
al momento de actualizar los valores Q, este valor determina la velocidad con la que el agente adapta su conocimiento 
conforme interactúa con el entorno.

Alfa alto: El agente prioriza fuertemente las experiencias más recientes al actualizar los valores de Q, realizando cambios 
rápidos y significativos en sus valores aprendidos, esto acelera el proceso de aprendizaje y permite adaptarse con rapidez, 
pero también puede provocar comportamientos inestables y variaciones constantes, ya que la información nueva tiende a 
sustituir gran parte del conocimiento previamente adquirido.

Alfa bajo: Las actualizaciones de los valores de Q se realizan de manera más lenta, dando menor importancia a cada 
nueva experiencia obtenida por el agente, debido a esto, el proceso de aprendizaje requiere una mayor cantidad de episodios 
para alcanzar la convergencia, sin embargo, los resultados obtenidos tienden a ser más estables, consistentes y menos propensos
a presentar cambios bruscos.

5. ¿Cuál de los dos algoritmos, SARSA o Q-learning, consideras que es más adecuado para el problema del blackjack y por qué?
Q-learning considero que es ser más adecuado para blackjack porque aprende directamente la mejor política posible, sin depender 
tanto de las acciones aleatorias de exploración, porque en comparación, el algoritmo de SARSA aprende siguiendo la política que 
el agente ejecuta en cada momento, mientras que la ventaja de Q-learning es que se enfoca en maximizar la recompensa futura. 
En este juego manejamos una cantidad de pocos movimientos por partida y requiere tomar decisiones óptimas rápidamente y Q-learning
obtiene mejores resultados en este entorno.

6. ¿Se puede explicar con cierta lógica del juego la política óptima encontrada por cada algoritmo? 
Sí, y la política aprendida coincide con el blackjack tradicional.

¿Qué acciones se toman en cada estado y por qué?
Sumas entre 4 y 11: La mejor acción es pedir otra carta, porque no hay riesgo de exceder 21 con una carta adicional.
Sumas entre 12 y 16: En esta situación la decisión depende de la carta visible del crupier. Si el crupier tiene una carta 
baja, conviene plantarse porque existe una alta posibilidad de que el crupier se pase, en cambio, si muestra una carta 
fuerte, es mejor pedir, ya que el crupier tiene más probabilidades de alcanzar una suma alta. 
Sumas de 17 a 19 sin as utilizable: La mejor acción sin duda es plantarse, debido a que la probabilidad de pasarse es 
elevada y la puntuación entra en juego.
Sumas de 17 y 18 con as utilizable: La mejor acción es pedir otra carta, porque el as puede cambiar su valor a 1 
y evitar que el jugador se pase de 21.
Sumas de 20 y 21: La mejor acción es plantarse, porque ya se cuenta con una mano muy fuerte y seguir pidiendo es un riesgo

****************************************************************************************
"""