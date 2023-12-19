import numpy as np
import matplotlib.pyplot as plt #plotting
from scipy.interpolate import interp1d #functional interpolation
from models import Mw


### value iteration functions

def value_it_fixedComp(s_grid_,
                       p_grid_,
                       outsidePolicy=0.5,
                       eta=0.1,
                       beta=0.95,
                       max_iterations=2000):

    """
    value iteration when competitors always charge a fixed price
    Args,
        s_grid (list): grid of states
        p_grid (list): grid of policies
    Returns:
        Dictionary of value and policy functions: {'value_function','policy_function'}
    """


    s_grid = np.array(s_grid_)
    p_grid = np.array(p_grid_)
    v_curr = np.ones(len(s_grid_))
    p_curr = np.ones(len(s_grid_), dtype=int)  # Initialize as integers for indexing

    for it in range(max_iterations):
        interp_v_func = interp1d(s_grid, v_curr, kind='linear', fill_value='extrapolate')

        new_weight_a = s_grid[:, np.newaxis] * (1 - eta * p_grid)
        new_weight_b = (1 - s_grid[:, np.newaxis]) * (1 - eta * outsidePolicy)
        new_state = new_weight_a / (new_weight_a + new_weight_b)
        future_u = beta * interp_v_func(new_state)

        values = p_grid[np.newaxis, :] * s_grid[:, np.newaxis] + future_u
        v_new = np.max(values, axis=1)
        p_curr = p_grid[np.argmax(values, axis=1)]

        if np.linalg.norm(v_new - v_curr) < 0.001:
            print('converged successfully in ' + str(it) + ' iterations')
            break
        else:
            v_curr = v_new

        if it == (max_iterations - 1):
            print('did not terminate after ' + str(it) + ' iterations')

    return {'value_function': v_curr.tolist(), 'policy_function': p_curr.tolist()}




def value_it_stratComp(s_grid,
                        p_grid,
                        eta=0.1,
                        beta=0.95,
                        max_iterations=2000):
    """
    Vectorized version of value_it_stratComp2
    Args:
        s_grid: grid of states
        p_grid: grid of policies
        nu: nu parameter
        beta: beta parameter
        max_iterations: maximum number of iterations
    Returns:
        Dictionary of value and policy functions: {'value_function', 'policy_function'}
    """
    s_grid = np.array(s_grid)
    p_grid = np.array(p_grid)

    v_curr = np.ones(len(s_grid))
    p_curr = np.ones(len(s_grid), dtype=int)  # Initialize as integers for indexing
    
    for it in range(max_iterations):
        interp_v_func = interp1d(s_grid, v_curr, kind='linear', fill_value='extrapolate')
        p_curr_b = p_curr[::-1]

        new_weight_a = s_grid[:, np.newaxis] * (1 - (eta * p_grid))
        new_weight_b = np.tile(np.multiply((1 - (eta * p_curr_b)),(1 - s_grid[:, np.newaxis]).T).T,len(p_grid))

        new_state = new_weight_a / (new_weight_a + new_weight_b)
        future_u = beta * interp_v_func(new_state)

        values = p_grid[np.newaxis, :] * s_grid[:, np.newaxis] + future_u
        v_new = np.max(values, axis=1)
        p_curr = p_grid[np.argmax(values, axis=1)]

        if np.linalg.norm(v_new - v_curr) < 0.001:
            print('converged successfully in ' + str(it) + ' iterations')
            break
        else:
            v_curr = v_new

        if it == (max_iterations - 1):
            print('did not terminate after ' + str(it) + ' iterations')

    return {'value_function': v_curr.tolist(), 'policy_function': p_curr.tolist()}



def value_it_stratComp_gen(s_grid,
                           p_grid,
                           eta=0.1,
                           beta=0.95,
                           max_iterations=2000,
                           starting_policy_1=None,
                           starting_policy_2=None,
                           starting_values_1=None,
                           starting_values_2=None):
    """
    Vectorized version of value_it_stratComp2
    Args:
        s_grid: grid of states
        p_grid: grid of policies
        eta: learning rate
        nu: nu parameter
        beta: beta parameter
        max_iterations: maximum number of iterations
    Returns:
        Dictionary of value and policy functions: {'value_function', 'policy_function'}
    """
    s_grid = np.array(s_grid)
    p_grid = np.array(p_grid)

    v_curr_1 = np.ones(len(s_grid)) if starting_values_1 is None else starting_values_1
    v_curr_2 = np.ones(len(s_grid)) if starting_values_2 is None else starting_values_2
    p_curr_1 = np.zeros(len(s_grid), dtype=int) if starting_policy_1 is None else starting_policy_1
    p_curr_2 = np.zeros(len(s_grid), dtype=int) if starting_policy_2 is None else starting_policy_2

    for it in range(max_iterations):
        ### seller 1 update
        interp_v_func_1 = interp1d(s_grid, v_curr_1, kind='linear', fill_value='extrapolate')
        p_curr_other = p_curr_2[::-1]

        new_weight_a = s_grid[:, np.newaxis] * (1 - (eta * p_grid))
        new_weight_b = np.tile(np.multiply((1 - (eta * p_curr_other)),(1 - s_grid[:, np.newaxis]).T).T,len(p_grid))

        new_state = new_weight_a / (new_weight_a + new_weight_b)
        future_u = beta * interp_v_func_1(new_state)

        values = p_grid[np.newaxis, :] * s_grid[:, np.newaxis] + future_u
        v_new_1 = np.max(values, axis=1)
        
        #update seller 1
        diff1 = np.linalg.norm(v_new_1 - v_curr_1) 
        p_curr_1 = p_grid[np.argmax(values, axis=1)]
        v_curr_1 = v_new_1

        ### seller 2 update
        interp_v_func_2 = interp1d(s_grid, v_curr_2, kind='linear', fill_value='extrapolate')
        p_curr_other = p_curr_1[::-1]

        new_weight_a = s_grid[:, np.newaxis] * (1 - (eta * p_grid))
        new_weight_b = np.tile(np.multiply((1 - (eta * p_curr_other)),(1 - s_grid[:, np.newaxis]).T).T,len(p_grid))

        new_state = new_weight_a / (new_weight_a + new_weight_b)
        future_u = beta * interp_v_func_2(new_state)

        values = p_grid[np.newaxis, :] * s_grid[:, np.newaxis] + future_u
        v_new_2 = np.max(values, axis=1)
        
        #update seller 2
        diff2 = np.linalg.norm(v_new_2 - v_curr_2) 
        p_curr_2 = p_grid[np.argmax(values, axis=1)]
        v_curr_2 = v_new_2

        if diff1+diff2 < 0.001:
            print('converged successfully in ' + str(it) + ' iterations')
            break

        if it == (max_iterations - 1):
            print('did not terminate after ' + str(it) + ' iterations')

    return {'value_function_1': v_curr_1.tolist(), 'policy_function_1': p_curr_1.tolist(),
            'value_function_2': v_curr_2.tolist(), 'policy_function_2': p_curr_2.tolist()}




def simulate_mw_bertrand(
    policy_function1,
    policy_function2,
    eta,
    rounds = 40,
    starting_share_player_1 = 0.1):
    """
    """
    
    price_transcript1 = [-1 for i in range(rounds)]
    price_transcript2 = [-1 for i in range(rounds)]
    state_transcript1 = [-1 for i in range(rounds)]
    state_transcript2 = [-1 for i in range(rounds)]
    state_transcript1[0] = starting_share_player_1
    state_transcript2[0] = 1-starting_share_player_1
    time_transcript = [1+i for i in range(rounds)]

    weights = np.array([starting_share_player_1,1-starting_share_player_1])


    #intialise multiplicate weights for consumer
    mw = Mw(weights = weights,eta_rate=eta)

    for r in range(rounds):
            share_player_1 = mw.weights[0] / sum(mw.weights)
            share_player_2 = mw.weights[1] / sum(mw.weights)
            price_player_1 = policy_function1(share_player_1)
            price_player_2 = policy_function2(share_player_2)
            price_vector = [price_player_1,price_player_2]
            mw.update(price_vector)
            #update arrays
            price_transcript1[r] = price_player_1
            state_transcript1[r] = share_player_1
            price_transcript2[r] = price_player_2
            state_transcript2[r] = share_player_2

    return( {'time': time_transcript,
             'price1' : price_transcript1,
             'price2' : price_transcript2,
             'share1' : state_transcript1,
             'share2' : state_transcript2
             })





### plotting funcs


def plot_valuefunc(states,values):
    plt.plot(states,values, label='Value function')
    plt.axvline(x=0.5, color='red', linestyle='--',alpha=0.5,linewidth=0.5)
    plt.xlabel('q')
    plt.ylabel('Future utility')
    plt.title('Seller value function')
    plt.legend()
    plt.show()


def plot_policyfunc(states,policies):
    plt.plot(states,policies, label='Policy function')
    plt.axvline(x=0.5, color='red', linestyle='--',alpha=0.5,linewidth=0.5)
    plt.xlabel('q')
    plt.ylabel('Price')
    plt.title('Seller policy function')
    plt.legend()
    plt.show()



def plot_price_dev(time,prices_player_1,prices_player_2,title='Price development'):
    plt.plot(time,prices_player_1, label='Seller 1')
    plt.plot(time,prices_player_2, label='Seller 2')
    plt.xlabel('Period')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_share_dev(time,shares_player_1,shares_player_2,title='Share development'):
    plt.plot(time,shares_player_1, label='Seller 1')
    plt.plot(time,shares_player_2, label='Seller 2')
    plt.xlabel('Period')
    plt.ylabel('Share')
    plt.title(title)
    plt.legend()
    plt.show()