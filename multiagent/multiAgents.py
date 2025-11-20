# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    Agente reflexo: escolhe uma ação avaliando apenas os estados sucessores imediatos.
    """

    def getAction(self, gameState: GameState):
        # Obtém a lista de ações legais disponíveis para o Pacman no estado atual.
        legalMoves = gameState.getLegalActions()
        # Para cada ação legal, calcula a avaliação do estado sucessor usando evaluationFunction.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # Encontra a maior pontuação entre os sucessores avaliados.
        bestScore = max(scores)
        # Encontra os índices de todas as ações que atingiram essa pontuação máxima (empates).
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # Escolhe aleatoriamente um dos índices com melhor pontuação — desempate estocástico.
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        # Retorna a ação correspondente ao índice escolhido (será executada pelo motor do jogo).
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        # Gera o estado sucessor resultante de aplicar a action ao estado atual.
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # Pega a posição do Pacman no estado sucessor (coordenadas x,y).
        newPos = successorGameState.getPacmanPosition()
        # Pega o objeto que representa a comida restante no sucessor (grid).
        newFood = successorGameState.getFood()
        # Pega o estado dos fantasmas (posições, timers de "scared", etc.) no sucessor.
        newGhostStates = successorGameState.getGhostStates()
        # Extrai os tempos de scared de cada fantasma (quanto tempo eles ficam vulneráveis).
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Inicializa a pontuação de avaliação com o score reportado pelo jogo no sucessor.
        score = successorGameState.getScore()

        # Para cada posição de fantasma no sucessor, calcula distância Manhattan até o Pacman.
        for ghostpos in successorGameState.getGhostPositions():
            dist = manhattanDistance(newPos, ghostpos)
            # Se um fantasma está muito próximo (< 2), penaliza fortemente para evitar colisão.
            if dist < 2:
                score -= 1000

        # Converte o grid de comida em lista de coordenadas onde ainda existe comida.
        foodList = newFood.asList()
        # Se não houver comida restante, recompensa fortemente porque é um estado desejável.
        if len(foodList) == 0:
            return score + 1000

        # Calcula a distância à comida mais próxima (Manhattan) — queremos aproximar-nos dela.
        closestFood = min(manhattanDistance(newPos, f) for f in foodList)
        # Dá um bónus que é inversamente proporcional à distância à comida mais próxima:
        # isto favorece movimentos que reduzem essa distância.
        score += 10.0 / closestFood

        # Penaliza a ação STOP para desencorajar o Pacman de ficar parado (risco / ineficiência).
        if action == Directions.STOP:
            score -= 50

        # Retorna a pontuação final do sucessor (maior = melhor).
        return score
def scoreEvaluationFunction(currentGameState: GameState):
    """
    Default evaluation function for adversarial search agents.
    Returns the game score for currentGameState (the built-in score).
    """
    return currentGameState.getScore()
class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax: agente adversarial que assume fantasmas perfeitamente adversariais (min).
    A implementação faz uma busca recursiva em árvore alternando agentes.
    """

    def getAction(self, gameState: GameState):
        # Função recursiva minimax que devolve (score, action) para o nó atual.
        def minimax(agentIndex, depth, gameState):
            # Se o estado for terminal (win/lose) ou a profundidade máxima for atingida,
            # usamos a função de avaliação para pontuar o estado e não expandimos mais.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            # Se o agente atual for o Pacman (index 0), queremos maximizar.
            if agentIndex == 0:
                bestScore = float('-inf')   # Começamos com -infinito (pior possível).
                bestAction = None
                # Iteramos sobre todas as ações legais do Pacman neste estado.
                for action in gameState.getLegalActions(agentIndex):
                    # Geramos o sucessor resultante desta ação.
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # Chamamos minimax para o próximo agente (1) mantendo a mesma profundidade.
                    score, _ = minimax(1, depth, successor)
                    # Se o valor retornado for melhor que o atual, atualizamos melhor score e ação.
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                # Devolvemos o melhor score encontrado e a ação correspondente.
                return bestScore, bestAction

            # Se o agente atual for um fantasma (index >= 1), queremos minimizar.
            else:
                # Define o índice do próximo agente e ajusta profundidade quando todos jogaram.
                nextAgent = agentIndex + 1
                nextDepth = depth
                # Se passarmos do último agente, voltamos ao Pacman (index 0) e incrementamos a profundidade.
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    nextDepth += 1

                bestScore = float('inf')  # Começa com +infinito (pior para o maximizador).
                bestAction = None
                # Para cada ação legal do fantasma, avaliamos o sucessor.
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # Recurse minimax para o próximo agente/nível.
                    score, _ = minimax(nextAgent, nextDepth, successor)
                    # Fantasma escolhe a ação que minimiza o score do Pacman.
                    if score < bestScore:
                        bestScore = score
                        bestAction = action
                return bestScore, bestAction

        # No nó raiz chamamos minimax com agentIndex 0 (Pacman) e depth 0.
        _, bestMove = minimax(0, 0, gameState)
        # Retornamos apenas a ação (o motor do jogo executa-a).
        return bestMove


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax com poda alfa-beta: igual ao Minimax, mas evita expandir ramos que não
    podem influenciar a decisão final, reduzindo nós expandidos.
    """

    def getAction(self, gameState: GameState):
        # Função recursiva que recebe alpha e beta para poda.
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            # Teste terminal: estado final ou profundidade máxima atingida.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            # Se for o Pacman (MAX), procuramos maximizar o valor.
            if agentIndex == 0:
                bestScore = float('-inf')
                bestAction = None
                # Para cada ação do Pacman:
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # Calcula valor recursivamente para o próximo agente com os limites alpha/beta.
                    score, _ = alphaBeta(1, depth, successor, alpha, beta)
                    # Se este sucessor é melhor do que o melhor conhecido, atualiza.
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                    # Se o melhor obtido já excede beta, podemos cortar o restante (poda beta).
                    if bestScore > beta:
                        break
                    # Atualiza alpha com a melhor pontuação encontrada até agora.
                    alpha = max(alpha, bestScore)
                return bestScore, bestAction

            # Se for um fantasma (MIN), procuramos minimizar o valor.
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
                # Quando se passa do último agente, volta ao Pacman e incrementa profundidade.
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    nextDepth += 1

                minVal = float('inf')
                bestAction = None
                # Para cada ação legal do fantasma:
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # Avalia recursivamente com os limites alpha/beta.
                    score, _ = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                    # Fantasma escolhe ação que reduz o score do Pacman (minimiza).
                    if score < minVal:
                        minVal = score
                        bestAction = action
                    # Se o mínimo atual é menor que alpha, podemos podar (poda alpha).
                    if minVal < alpha:
                        break
                    # Atualiza beta com o menor valor encontrado até agora.
                    beta = min(beta, minVal)
                return minVal, bestAction

        # Chamada na raiz: inicializa alpha/beta e percorre ações do Pacman.
        _, action = alphaBeta(0, 0, gameState, float('-inf'), float('inf'))
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax: similar ao Minimax, mas modela fantasmas como agentes estocásticos.
    Em vez de min (pior caso) usa-se a esperança (média) sobre ações dos fantasmas.
    """

    def getAction(self, gameState: GameState):
        # Função recursiva que retorna (valor, ação) para o nó atual.
        def expectimax(agentIndex, depth, gameState):
            # Se o estado for terminal ou a profundidade limite for atingida, avalia o estado.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            # Lista de ações legais para o agente atual.
            legalActions = gameState.getLegalActions(agentIndex)
            # Se não houver ações legais, devolve a avaliação do estado corrente.
            if len(legalActions) == 0:
                return self.evaluationFunction(gameState), None

            # Se for o Pacman (MAX), escolhe a ação com maior valor esperado.
            if agentIndex == 0:
                bestValue = float('-inf')
                bestAction = None
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # Avalia o valor do sucessor assumindo o próximo agente age segundo expectimax.
                    value, _ = expectimax(1, depth, successor)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction

            # Se for um fantasma (chance node), calcula o valor esperado (média uniforme).
            else:
                nextIndex = agentIndex + 1
                nextDepth = depth
                # Se passarmos do último agente, voltamos ao Pacman e incrementamos a profundidade.
                if nextIndex == gameState.getNumAgents():
                    nextIndex = 0
                    nextDepth = depth + 1

                probability = 1.0 / len(legalActions)  # Modelamos escolhas dos fantasmas como uniformes.
                expectedValue = 0
                # Soma ponderada dos valores dos sucessores (esperança).
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(nextIndex, nextDepth, successor)
                    expectedValue += probability * value
                # Retorna o valor esperado; não há uma ação "ótima" do fantasma para retornar aqui.
                return expectedValue, None

        # Chamada na raiz: devolve a ação do Pacman que maximiza o valor esperado.
        _, bestAction = expectimax(0, 0, gameState)
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Versão ajustada da betterEvaluationFunction para melhorar a média de score.
    Principais mudanças:
    - maior peso para proximidade da comida (favorece rotas rápidas para pellets)
    - penalidade por comida menos agressiva (evita caminhos subóptimos)
    - maior recompensa por cápsulas e por comer fantasmas assustados (pontos grandes)
    - mantém penalidades fortes para colisões/fantasmas perigosos
    """
    # Terminal checks
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')

    # Base score fornecido pelo jogo
    score = currentGameState.getScore()

    # Info do estado
    pacPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    numFood = len(foodList)
    capsules = currentGameState.getCapsules()
    numCapsules = len(capsules)
    ghostStates = currentGameState.getGhostStates()

    # 1) Penalidade/ incentivo relacionado à comida
    # Reduzimos a penalidade por comida restante (de -14 para -8)
    # para evitar forçar rotas que sacrificam pontos por segurança.
    score += -8.0 * numFood

    # Aumentamos fortemente o bónus por proximidade à comida (de 20 para 50),
    # usando recíproco para priorizar comida próxima — isto acelera consumo de pellets.
    if foodList:
        distsToFood = [manhattanDistance(pacPos, f) for f in foodList]
        closestFood = min(distsToFood)
        score += 50.0 / (closestFood + 1)
    else:
        # Bónus grande se não houver comida
        score += 2000.0

    # 2) Cápsulas (power pellets)
    # Aumentamos recompensa por pegar cápsula (de 150 para 300) pois cápsulas
    # permitem comer fantasmas e obter muitos pontos.
    if pacPos in capsules:
        score += 300.0
    # Mantemos penalidade por cápsulas restantes para incentivar a recolha.
    score += -20.0 * numCapsules

    # 3) Fantasmas: perigo e oportunidades
    # Penaliza proximidade a fantasmas não-assustados (sobrevivência),
    # e recompensa perseguir fantasmas assustados se for possível com o tempo restante.
    minActiveGhostDist = float('inf')
    for gs in ghostStates:
        ghostPos = gs.getPosition()
        ghostDist = manhattanDistance(pacPos, ghostPos)

        if gs.scaredTimer > 0:
            # Se o fantasma está assustado e o tempo de scared é suficiente para alcançá-lo,
            # damos grande incentivo proporcional à distância e ao tempo restante.
            # Isto encoraja o Pacman a perseguir fantasmas que rendem muitos pontos.
            if ghostDist > 0:
                # peso maior se o scaredTimer permite alcançar: (scaredTimer / dist)
                score += 200.0 * (gs.scaredTimer / (ghostDist + 1))
        else:
            # Fantasma ativo: penalidades robustas para evitar colisões/adjacência.
            minActiveGhostDist = min(minActiveGhostDist, ghostDist)
            if ghostDist == 0:
                score -= 5000.0
            elif ghostDist <= 2:
                score -= 1000.0 / ghostDist
            elif ghostDist <= 5:
                score -= 40.0 / ghostDist

    # 4) Pequena recompensa por comer clusters de comida:
    # estimativa rápida: distância média aos 3 alimentos mais próximos (se existirem)
    # adiciona incentivo a mover‑se para regiões densas de comida.
    if len(foodList) > 0:
        dists_sorted = sorted([manhattanDistance(pacPos, f) for f in foodList])
        top_k = dists_sorted[:3]  # 3 comidas mais próximas
        avg_dist = float(sum(top_k)) / len(top_k)
        # bónus proporcional ao inverso da distância média às 3 mais próximas
        score += 30.0 / (avg_dist + 1)

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
