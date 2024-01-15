import numpy as np
from matplotlib import pyplot as plt


def p_measure(z, x):
    """Looks up measurement probability.
    returns p(z|x)
    Args:
        z: measurement 0 = closed, 1 = open. int
        x: initial state, 0 = closed, 1 = open. int
    Returns:
        p: probability of z given x. float
    """

    if [z, x] == [1, 1]:
        p = 0.6
    elif [z, x] == [0, 1]:
        p = 0.4
    elif [z, x] == [1, 0]:
        p = 0.2
    elif [z, x] == [0, 0]:
        p = 0.8
    return p

def p_transition(X, u, x):
    """Looks up state transition probability.
    returns p(X|u,x)
    Args:
        X: state we want belief from, 0 = closed, 1 = open. int
        u: robot action for current step, 0 = do nothing, 1 = push. int
        x: initial state, 0 = closed, 1 = open. int
    Returns:
        po: probability door is closed. flaot
        p1: probability door is open. flaot
    """

    if [x, u] == [0, 0]: # closed, do nothing
        p0 = 1.0
    elif [x, u] == [1, 0]:  # open, do nothing
        p0 = 0.0
    elif [x, u] == [1, 1]:  # open, push
        p0 = 0.0
    elif [x, u] == [0, 1]:  # closed, push
        p0 = 0.2

    p1 = 1.0 - p0

    if X:
        return p1
    else:
        return p0

def predict(bel, u, debug):
    """Implements predict step for Bayes Filter algotithm.
    Args:
        bel: prior belief. [float, float]
        u: robot action for current step. int
        debug: print out debug statements if true. boolean
    Returns:
        bel: Posterior belief. [float, float]
    """

    bel2  = [0, 0]
    bel2[0] = p_transition(0, u, 1) * bel[1] + p_transition(0, u, 0) * bel[0]
    bel2[1] = p_transition(1, u, 1) * bel[1] + p_transition(1, u, 0) * bel[0]
    bel = bel2

    if debug:
        print('Posterior belief: ' + str(np.round(bel, 4)))

    return bel

def update(bel, z, debug):
    """Implements update step for Bayes Filter algotithm.
    Args:
        bel: posterior belief. [float, float]
        z: measurement for current step, 0 = closed, 1 = open. int
        debug: print out debug statements if true. boolean
    Returns:
        bel: Updated belief. [float, float]
    """
    bel[0] = p_measure(z, 0) * bel[0]
    bel[1] = p_measure(z, 1) * bel[1]
    if debug:
        print('+ Measurement belief: ' + str(np.round(bel, 4)))

    # normalize bel so sum(bel) = 1.0
    n = sum(bel)
    bel = np.divide(bel, n)

    # if debug:
    #     print('+ normalization belief: ' + str(np.round(bel, 4)) + ' n = ' + str(round(n, 4)))
    if debug:
        print('Updated belief: ' + str(np.round(bel, 4)))

    return bel

def Bayes(u, z, bel = [0.5, 0.5], debug = False, plot = False):
    """Performs Bayes filtering
    Args:
        bel: Initial beliefs, default to [0.5, 0.5]. [float, float]
        u: robot actions, 0 = do nothing, 1 = push. nx1 int array
        z: measurements, 0 = closed, 1 = open. nx1 int array
        debug: print out debug statements if true, default off. boolean
        plot: option to plot bel at each iteration, default off. boolean
    Returns:
        bels: history of beliefs. nx2 float array
    """

    # Ensures u and z are of same length
    if len(u) != len(z):
        raise Exception("u and z must be same length")

    # init bels
    bels = np.zeros([len(u)+1, 2])
    bels[0,:] = bel

    for k in range(len(u)): # iterate through steps and predict/update bel
        if debug:
            print('***** Iteration '+ str(k) +' *****')
        bel = predict(bel, u[k], debug)
        bel = update(bel, z[k], debug)
        bels[k+1,:] = bel
    
    if plot:    # Make bel vs iteration plot
        plt.plot(np.arange(len(u)+1), bels[:,0], marker='o', label='bel(Closed)')
        plt.plot(np.arange(len(u)+1), bels[:,1], marker='o', label='bel(Open)')
        plt.grid(True)
        plt.legend()
        plt.show()

    return bels

def main():
    """
    Main code block that runs Bayes() function for all use cases
    """
    ## Question 1 - action = do nothing, measurement = door open
    u =  np.zeros([50,1]) # define actions 
    z = np.ones([50,1]) # define measurements
    bels = Bayes(u, z, debug = False, plot = False) # run Bayes filter
    print('Q1) The robot will be 99.99% certain that the door is open after iteration ' + str(np.argmax(bels[:,1] > 0.9999)))

    ## Question 2 - action = do nothing, measurement = door closed
    u =  np.ones([50,1]) # define actions
    z = np.ones([50,1]) # define measurements
    bels = Bayes(u, z, debug = False, plot = False) # run Bayes filter
    print('Q2) The robot will be 99.99% certain that the door is open after iteration ' + str(np.argmax(bels[:,1] > 0.9999)))

    ## Question 3 - action = push, measurement = door closed
    u =  np.ones([50,1]) # define actions
    z = np.zeros([50,1]) # define measurements
    bels = Bayes(u, z, debug = False, plot = False) # run Bayes filter

    # Current estimate is highest prop belief
    if bels[-1,1] > 0.5:
        ssbelief = 'Open'
        certainty = 100 * bels[-1,1]
    else:
        ssbelief = 'Closed'
        certainty = 100 * bels[-1,0]

    print('Q3) The steady state belief is ' + ssbelief + ' with a certainty of ' + str(round(certainty, 5)) + '%')

if __name__ == '__main__':
    main()
