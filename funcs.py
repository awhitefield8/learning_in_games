import numpy as np
import matplotlib.pyplot as plt #plotting
from scipy.interpolate import interp1d #functional interpolation





def inverse_mw(new_weight,old_weight,nu=0.1):
    """
    Args:
        new_weight: new weight
        old_weight: initial weight
        cost: cost
        nu: learning rate
    Returns:
        cost required to generate new weight
    """
    return (1 - (new_weight/old_weight))/nu



### value iteration functions

def value_it_fixedComp(s_grid,
                       p_grid,
                       outsidePolicy = 0.5,
                       nu = 0.2,
                       beta = 0.95,
                       max_iterations = 200):
    """
    value iteration when competitors always charge a fixed price
    Args,
        s_grid: grid of states
        p_prid: grid of policies
    Returns:
        Dictionary of value and policy functions: {'value_function','policy_function'}
    """

    v_curr = [1]*len(s_grid)
    p_curr = [1]*len(s_grid) #policy function ('grid')



    ### idea: start with a grid of states. generate
    # i) an interpolated value function (for the player to estimte future values)
    # ii) an interpolated policy function (to simulate the other players play)


    for it in range(max_iterations):
        #loop through states, and for each, pick an optimal policy, and update corresponding value
        v_new = [-1]*len(v_curr) #prep new list
        # define functions for analysis
        interp_v_func = interp1d(s_grid, #grid points
                                v_curr, #values
                                kind='linear', fill_value='extrapolate')
        for j in range(len(s_grid)):
            s = s_grid[j]
            values = [-1]*len(p_grid)
            for i in range(len(p_grid)):
                p = p_grid[i]
                flow_u = p*s
                new_weight_a = s*(1 - (nu*p))
                new_weight_b = (1-s)*(1 - (nu*outsidePolicy))
                new_state = new_weight_a / (new_weight_a + new_weight_b)
                future_u = beta*interp_v_func(new_state)
                ### interpolating function
                values[i]= flow_u + future_u
            v_new[j] = np.array(values).max()
            p_curr[j] = p_grid[np.array(values).argmax()]    ### break ties by picking lowest price. policy correspondence is not convex. >>> could extend by breaking randomly

        if np.linalg.norm(np.array(v_new) - np.array(v_curr)) < 0.001:
            print('coverged successfully in ' + str(it) + ' iterations')
            break
        else:
            v_curr = v_new

        if it == (max_iterations - 1):
            print('did not terminate after ' + str(it) + ' iterations')

    return({'value_function': v_curr,'policy_function': p_curr})




### plotting funcs


def plot_valuefunc(states,values):
    plt.plot(states,values, label='Value function')
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.xlabel('q')
    plt.ylabel('Future utility')
    plt.title('Firm value function')
    plt.legend()
    plt.show()


def plot_policyfunc(states,policies):
    plt.plot(states,policies, label='Policy function')
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.xlabel('q')
    plt.ylabel('Price')
    plt.title('Firm policy function')
    plt.legend()
    plt.show()



def plot_price_dev(time,prices_player_1,prices_player_2):
    plt.plot(time,prices_player_1, label='Player 1')
    plt.plot(time,prices_player_2, label='Player 2')
    plt.xlabel('Period')
    plt.ylabel('Price')
    plt.title('Price development')
    plt.legend()
    plt.show()

def plot_share_dev(time,shares_player_1,shares_player_2):
    plt.plot(time,shares_player_1, label='Player 1')
    plt.plot(time,shares_player_2, label='Player 2')
    plt.xlabel('Period')
    plt.ylabel('Share')
    plt.title('Share development')
    plt.legend()
    plt.show()