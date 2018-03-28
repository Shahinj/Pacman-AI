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
import numpy as np
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        "*** YOUR CODE HERE ***"
        curr_food = currentGameState.getFood()
        
        # prev_pos = currentGameState.getPacmanPosition()
        sum = 0
        dist_food = None
        prev_dist_food = 0
        for j in range(successorGameState.getWalls().height):
            for i in range(successorGameState.getWalls().width):
                if curr_food[i][j] == True:
                    if dist_food == None or dist_food >= manhattanDistance((i,j),newPos):
                        dist_food = manhattanDistance((i,j),newPos)
        # print dist_food
        if dist_food != None:
            sum -= dist_food
        
        for i in newGhostStates:
            if (manhattanDistance(i.getPosition(),newPos) > 1):
                sum += 0
            else:
                sum += -50

        if (curr_food[newPos[0]][newPos[1]] == True):
            sum += 10
        
        if successorGameState.isWin():
            sum += 1000

        return sum
        
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        
        def DFMiniMax(state, player,depth):
                        
            if state.isWin() == True or state.isLose() == True or depth >= self.depth:
                return (self.evaluationFunction(state),None)
            else:
                actions = state.getLegalActions(player)
                values = {}
                for action in actions:
                    child_state = state.generateSuccessor(player,action)
                    next_player = (player+1) % state.getNumAgents();
                    if next_player == 0:
                        value_action = DFMiniMax(child_state,next_player,depth+1)[0]
                    else:
                        value_action = DFMiniMax(child_state,next_player,depth)[0]
                    values[action] = value_action
                
                if player == 0:
                    return max(values.values()),[action for action in actions if values[action] == max(values.values())]
                else:
                    return min(values.values()),[action for action in actions if values[action] == min(values.values())]

        action = DFMiniMax(gameState,self.index,0)[1][0]
        return action
                        
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def AlphaBeta(state, player,alpha,beta,depth):
                        
            if state.isWin() == True or state.isLose() == True or depth >= self.depth:
                return (self.evaluationFunction(state),None)
            else:
                actions = state.getLegalActions(player)
                values = {}
                child_list = []
                next_player = (player+1) % state.getNumAgents();
                opt_action = None
                if (next_player == 0):
                    inc_depth = 1
                else:
                    inc_depth = 0
                
                if player == 0:
                    for action in actions:
                        c = state.generateSuccessor(player,action)
                        alpha2 = AlphaBeta(c,next_player,alpha,beta,depth+inc_depth)[0]
                        if (opt_action == None):
                            opt_action = action
                        if (alpha2 > alpha):
                            opt_action = action
                            alpha = alpha2       
                        if beta <= alpha:
                            break
                    return (alpha,opt_action)
                else:
                    for action in actions:
                        c = state.generateSuccessor(player,action)
                        beta2 = AlphaBeta(c,next_player,alpha,beta,depth+inc_depth)[0]
                        if (opt_action == None):
                            opt_action = action
                        if (beta2 < beta):
                            opt_action = action
                            beta = beta2
                        if beta <= alpha:
                            break
                    return (beta,opt_action)

        action = AlphaBeta(gameState,self.index,-1*float('inf'),float('inf'),0)[1]
        return action
                        
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def DFMiniMax(state, player,depth):
                        
            if state.isWin() == True or state.isLose() == True or depth >= self.depth:
                return (self.evaluationFunction(state),None)
            else:
                actions = state.getLegalActions(player)
                values = {}
                for action in actions:
                    child_state = state.generateSuccessor(player,action)
                    next_player = (player+1) % state.getNumAgents();
                    if next_player == 0:
                        value_action = DFMiniMax(child_state,next_player,depth+1)[0]
                    else:
                        value_action = DFMiniMax(child_state,next_player,depth)[0]
                    values[action] = value_action
                
                if player == 0:
                    return max(values.values()),[action for action in actions if values[action] == max(values.values())]
                else:
                    return (sum(values.values())/float(len(values.values())),None)

        action = DFMiniMax(gameState,self.index,0)[1][0]
        return action
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    win = 1000
    lose = -1000
    food = 1
    pellet = 10
    sum = 0
    
    state = currentGameState
    curr_food = currentGameState.getFood()
    pellets = state.getCapsules()
    newGhostStates = state.getGhostPositions()
    pos = state.getPacmanPosition()
    
    if state.isWin() or state.getNumFood() == 0:
        # print 'win'
        sum += 500
        # return sum
    elif state.isLose():
        sum += 500
        # print 'lose'
        # return sum
    
    
    # sum += 50*np.tanh(currentGameState.getNumFood()/1000.0)    
    

    counter_ghost = 0
    min_dist_ghost = float('inf')
    for i in newGhostStates:
        dist = manhattanDistance(i,pos)
        if (dist < min_dist_ghost):
            min_dist_ghost = dist
    
    # if min_dist_ghost != float('inf'):
    #     sum += 100*np.tanh(min(3,min_dist_ghost)/1000.0)
    
    # min_dist_food = float('inf')
    # min_pos = None
    # food_list = []
    # for j in range(state.getWalls().height):
    #     for i in range(state.getWalls().width):
    #         if curr_food[i][j] == True:
    #             food_list.append((i,j))
    #             dist_food = manhattanDistance((i,j),pos)
    #             if dist_food < min_dist_food:
    #                 min_dist_food = dist_food
    #                 min_pos = (i,j)
    
    
    avg_dist_food = 0
    min_pos = None
    cnt = 0
    food_list = []
    for j in range(state.getWalls().height):
        for i in range(state.getWalls().width):
            if curr_food[i][j] == True:
                food_list.append((i,j))
                avg_dist_food += manhattanDistance((i,j),pos)
                cnt += 1.0
    
    # sum += np.tanh(-1*min_dist_food/1000.0)
    # sum += np.tanh(scoreEvaluationFunction(state)/1000.0)*500
    # sum += state.getScore()
    sum += -1*state.getNumFood() - 1 * (min(3,min_dist_ghost)-5) + 3*state.getScore()
    # sum += -1*state.getNumFood() - 2 * (1.0/(min_dist_ghost+0.1)) + 1.5*state.getScore()
        
    if cnt != 0:
        sum +=  -0.5*avg_dist_food/cnt


    # print sum
    return sum
    
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

