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
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # Initialize score from successor's game score (baseline)
        score = float(successorGameState.getScore())
        foodList = newFood.asList()
        numFood = len(foodList)
        # Capsules (power pellets)
        capsules = successorGameState.getCapsules()
        numCapsules = len(capsules)

        # Ghost info and scared timers
        ghostStates = successorGameState.getGhostStates()
        ghostPositions = [gs.getPosition() for gs in ghostStates] if ghostStates else []

        # Penalidade proporcional ao número de comidas restantes (incentiva comer)
        score += -4.0 * numFood

        # Bónus pela comida mais próxima: mais alto quanto mais perto estiver
        if foodList:
            # calcula distâncias Manhattan até todas as comidas e pega a mínima
            dists = [manhattanDistance(newPos, food) for food in foodList]
            closestFoodDist = min(dists)
            # bónus inversamente proporcional (evita divisão por zero)
            score += 10.0 / (closestFoodDist + 1)
        else:
            # se não há comida, adiciona um grande bónus (estado ótimo)
            score += 500.0

        # Cápsulas: bónus por estar em cima de uma cápsula e penalidade por tê-las remanescentes
        if newPos in capsules:
            score += 50.0
        score += -2.0 * len(capsules)

        # Avalia proximidade de fantasmas:
        # - fantasmas não assustados: penaliza fortemente proximidade
        # - fantasmas assustados: incentiva perseguição se seguro
        for idx, gs in enumerate(ghostStates):
            gpos = gs.getPosition()
            dist = manhattanDistance(newPos, gpos)
            scared = gs.scaredTimer > 0

            if scared:
                # pequeno incentivo para perseguir fantasmas assustados (decrescente com a distância)
                if dist > 0:
                    score += 8.0 / dist
            else:
                # fantasma perigoso: penalidades escalonadas para reduzir mortes
                if dist == 0:
                    # no mesmo quadrado -> morte imediata (muito ruim)
                    score -= 1000.0
                elif dist <= 1:
                    # muito perto -> evite fortemente
                    score -= 200.0
                elif dist <= 3:
                    # perto -> penalidade considerável, menor a medida que aumenta a distância
                    score -= 40.0 / dist
                else:
                    # longe -> nenhuma penalidade extra
                    score -= 0.0

        # Penaliza permanecer parado (ação STOP) para evitar inércia
        if action == Directions.STOP:
            score -= 5.0

        # Incentivo adicional para aproximar-se de comida quando não há perigo imediato
        # Verifica a menor distância a um fantasma para julgar segurança
        ghostDistances = [manhattanDistance(newPos, gpos) for gpos in ghostPositions] if ghostPositions else [float('inf')]
        minGhostDist = min(ghostDistances)
        if minGhostDist > 2 and foodList:
            # dá um impulso maior à proximidade da comida quando é seguro fazê-lo
            score += 5.0 / (closestFoodDist + 1) if foodList else 0.0

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(agent, depth, state):
            if state.isWin() or state.isLose() or (depth == self.depth and agent == 0):
                return self.evaluationFunction(state)
            numAgents = state.getNumAgents()
            nextAgent = (agent + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getLegalActions(agent)
            if not actions:
                return self.evaluationFunction(state)

            if agent == 0:  # Pacman (max)
                return max(minimax(nextAgent, nextDepth, state.generateSuccessor(agent, action)) for action in actions)
            else:           # Fantasmas (min)
                return min(minimax(nextAgent, nextDepth, state.generateSuccessor(agent, action)) for action in actions)

        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            value = minimax(1, 0, gameState.generateSuccessor(0, action))
            if value > bestScore:
                bestScore = value
                bestAction = action
        return bestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta(agent, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or (depth == self.depth and agent == 0):
                return self.evaluationFunction(state)
            numAgents = state.getNumAgents()
            nextAgent = (agent + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getLegalActions(agent)
            if not actions:
                return self.evaluationFunction(state)

            if agent == 0:  # Pacman (max)
                v = float('-inf')
                for action in actions:
                    v = max(v, alphabeta(nextAgent, nextDepth, state.generateSuccessor(agent, action), alpha, beta))
                    if v > beta: return v         # Poda beta
                    alpha = max(alpha, v)
                return v
            else:           # Fantasmas (min)
                v = float('inf')
                for action in actions:
                    v = min(v, alphabeta(nextAgent, nextDepth, state.generateSuccessor(agent, action), alpha, beta))
                    if v < alpha: return v        # Poda alpha
                    beta = min(beta, v)
                return v

        bestScore = float('-inf')
        bestAction = Directions.STOP
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            value = alphabeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if value > bestScore:
                bestScore = value
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(agent, depth, state):
            if state.isWin() or state.isLose() or (depth == self.depth and agent == 0):
                return self.evaluationFunction(state)
            numAgents = state.getNumAgents()
            nextAgent = (agent + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getLegalActions(agent)
            if not actions:
                return self.evaluationFunction(state)

            if agent == 0:  # Pacman (max)
                return max(expectimax(nextAgent, nextDepth, state.generateSuccessor(agent, action)) for action in actions)
            else:           # Fantasmas (expectation)
                total = 0
                for action in actions:
                    prob = 1.0 / len(actions)
                    total += prob * expectimax(nextAgent, nextDepth, state.generateSuccessor(agent, action))
                return total

        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            value = expectimax(1, 0, gameState.generateSuccessor(0, action))
            if value > bestScore:
                bestScore = value
                bestAction = action
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
