"""
In this script we implement the AHPI algorithm.
"""

import logging
import pandas       as pd
import numpy        as np
import scipy.stats  as stats

from scipy.optimize import fsolve


def AHPI(df, MII=50, MIO=50, minimum_iterations=10, convergence_threshold=0.01, fit_valence_prob=True,
         fit_privilege=True):
    '''
    The AHPI (=asymmetric heterogenous pariwise interactions) algorithm uses a generalized Bradley-Terry model to fit
    scores, valence probabilities and privileges based on pairwise interactions of the form 'privileged individual
    (e.g. a defendant), unprivileged individual (e.g. a plaintiff), winner, interaction_type (e.g. the case type of a 
    trial)'. The model is fitted using an Expectation-Maximization algorithm.
    The algorithm iteratively updates all the fitted values (outer loop) and in each iteration updates the fitted
    values. The fitting of the scores is done iteratively (inner loop).


    :param df:                      Dataframe of pairwise interactions with columns 'priv', 'unpriv', 'win_index' (int),
                                    'val_type', 'priv_type' representing the privileged individual, unprivileged
                                    individual, index of winning individual (0=winner privileged or 1 else), valence
                                    type, privilege type
    :param minimum_iterations:      Minimum iterations in inner and outer loop
    :param MII:                     Maximum iterations in inner loop
    :param MIO:                     Maximum iterations in outer loop
    :param convergence_threshold:   Convergence threshold for scores, valence probabilities, privileges
    :param fit_valence_prob:        Boolean, determinig if the valence probabilities should be fitted or set to 1
    :param fit_privilege:           Boolean, determinig if the privileges should be fitted or set to 0
    :return:                        Dictionaries of exponentials of the scores, valence probabilities, privileges
    '''

    df = df.copy()      # Create a copy of the DataFrame to avoid modifying the original one

    # create mappings: the assigned index will be the index also used when calling fitted scores, valence probabilities,
    # privileges
    ####################################################################################################################
    indiv_map     = {value: idx for idx, value in enumerate(pd.concat([df['priv'], df['unpriv']]).unique())}
    val_type_map  = {value: idx for idx, value in enumerate(df['val_type'].unique())}
    priv_type_map = {value: idx for idx, value in enumerate(df['priv_type'].unique())}

    # create repositories for fitted ln scores, valence probabilities, privileges
    ####################################################################################################################
    exp_scores  = np.full(len(indiv_map),0.9)                                        # array for ln scores
    val_probs   = np.full(len(val_type_map),0.5) if fit_valence_prob \
             else np.full(len(val_type_map),1.0)                                     # array for valence probs
    privileges  = np.full(len(priv_type_map),0.0)                                    # array for privileges

    # Map the individuals, valence types and privilege types
    ####################################################################################################################
    df['priv']        = df['priv'].map(indiv_map)
    df['unpriv']      = df['unpriv'].map(indiv_map)
    df['val_type']    = df['val_type'].map(val_type_map)
    df['priv_type']   = df['priv_type'].map(priv_type_map)

    # assign u (winning individual), v (losing individual), c for latter computations
    ####################################################################################################################
    df['u'] = np.where(df['win_index'] == 0, df['priv'],df['unpriv'])
    df['v'] = np.where(df['win_index'] == 1, df['priv'],df['unpriv'])
    df['c'] = np.where(df['win_index'] == 0, -1, 1)
    df.drop(columns=['priv', 'unpriv'], inplace=True)

    # initialise fitted valence probabilities and privileges with initial guesses
    ####################################################################################################################
    df['q']   = val_probs[df['val_type']]
    df['eps'] = privileges[df['priv_type']]

    class ConvergenceChecker:
        '''
        class checking the convergence for inputs: initialised with maximum_iterations allowed and convergence_threshold
        being the maximum absolute difference in subsequent iterations
        updated via update(): current_lambda, current_epsilon, current_q
        '''
        def __init__(self, maximum_iterations, minimum_iterations = minimum_iterations,
                     convergence_threshold = convergence_threshold):
            self.maximum_iterations     = maximum_iterations                    # Store the maximum number of iterations
            self.minimum_iterations     = minimum_iterations                    # Store the minimum number of iterations
            self.convergence_threshold  = convergence_threshold                 # Store the convergence threshold
            self.old_lambdas            = []                                    # Initialize list for past lambda's
            self.old_epsilons           = []                                    # Initialize list for past epsilon's
            self.old_q_s                = []                                    # Initialize list for past q's
            self.loop_number            = 0                                     # Initialize loop counter

        def update(self, current_lambda, current_epsilon, current_q):  # Update method to check for convergence
            '''
            Takes current_lambda, current_epsilon, current_q as inputs.
            -updates loop number, old_lambdas, old_epsilons, old_q_s
            -test if:   1. loop_number > maximum_iterations
                        2. Kendall correlation not changing and larger than 0.999
                        3. max abs difference in lambdas, epsilons, privileges smaller than convergence_threshold
            :return:    -if 1. or 2.+3.: 0, loop_number
                        -else:           1, loop_number
            '''
            self.loop_number += 1                                       # Increment the loop counter

            if self.loop_number > self.maximum_iterations: return 0, self.loop_number # case: maximum iterations

            self.old_lambdas.append(np.copy(current_lambda))        # Append lambdas
            self.old_epsilons.append(np.copy(current_epsilon))      # Append epsilons
            self.old_q_s.append(np.copy(current_q))                 # Append q_s

            if len(self.old_lambdas) > 3:  # Keep only the last three lambda, epsilon and q values
                self.old_lambdas.pop(0), self.old_epsilons.pop(0), self.old_q_s.pop(0)

            if self.loop_number < self.minimum_iterations: return 1, self.loop_number  # case: minimum iterations

            if len(self.old_lambdas) >= 3:  # Check if there are at least 3 iterations
                kendall_corr = stats.kendalltau(self.old_lambdas[-1], self.old_lambdas[-2])[0]

                if kendall_corr > 0.999:  # Kendall_corr>0.999 and not changing
                    max_abs_diff_lambda     = max(abs(np.subtract(self.old_lambdas[-1], self.old_lambdas[-2])))
                    max_abs_diff_epsilon    = max(abs(np.subtract(self.old_epsilons[-1], self.old_epsilons[-2])))
                    max_abs_diff_q          = max(abs(np.subtract(self.old_q_s[-1], self.old_q_s[-2])))

                    if (max_abs_diff_lambda      < self.convergence_threshold and
                            max_abs_diff_epsilon < self.convergence_threshold and
                            max_abs_diff_q       < self.convergence_threshold):  # Check if all differences < threshold
                        return 0, self.loop_number  # Return if convergence criteria met

            return 1, self.loop_number  # Return the updated state

    card_q_t        = df['val_type'].value_counts().sort_index().values
    outer_checker   = ConvergenceChecker(maximum_iterations=MIO)

    # Precompute indices for each individual for computational speed
    precomputed_u = {idx: df.index[df['u'] == idx].to_numpy() for idx in range(len(exp_scores))}
    precomputed_v = {idx: df.index[df['v'] == idx].to_numpy() for idx in range(len(exp_scores))}

    logging.info(f'Starting with the optimisation.')
    while True:   

        # assign current lambda of scores to u (winning individual), v (losing individual),
        ################################################################################################################
        df['lambda_u']    = exp_scores[df['u']]
        df['lambda_v']    = exp_scores[df['v']]
        
        # calculate pi
        ################################################################################################################
        df['pi'] = np.exp(df['c'] * df['eps'])    * df['lambda_u'] * df['q'] \
                         / (df['lambda_u'] * np.exp(df['c'] * df['eps']) * df['q'] +
                            df['lambda_v'] * (1 - df['q']))
        
        # fit valence probability
        ################################################################################################################
        if fit_valence_prob:
            for idx in range(len(val_probs)):
                val_probs[idx] = df.loc[df['val_type'] == idx, 'pi'].sum() / card_q_t[idx]
            df['q'] = val_probs[df['val_type']]
        
        # fit privileges
        ################################################################################################################
        if fit_privilege:
            for idx in range(len(privileges)):
                mask   = df['priv_type'] == idx
                df_idx = df[mask]
                pi_idx, c_idx,lambda_u_idx,lambda_v_idx = \
                    df_idx['pi'], df_idx['c'],  df_idx['lambda_u'], df_idx['lambda_v']
                def func_epsilon(x):
                    y = (1 - np.exp(x)) / (1 + np.exp(x))
                    y += np.sum(pi_idx * c_idx - lambda_u_idx * np.exp(c_idx * x) * c_idx
                                 / (lambda_u_idx * np.exp(c_idx * x) + lambda_v_idx))
                    return y

                privileges[idx] = fsolve(func_epsilon, 0.0)[0]

            df['eps'] = privileges[df['priv_type']]
        
        # fit scores
        ################################################################################################################
        inner_checker = ConvergenceChecker(maximum_iterations=MII)
        while True:
            for idx in range(len(exp_scores)):
                df_u_r          = df.loc[precomputed_u[idx]]
                df_v_r          = df.loc[precomputed_v[idx]]
                # convert to arrays
                win_index_u_r   = df_u_r['win_index'].values
                eps_u_r         = df_u_r['eps'].values
                pi_u_r          = df_u_r['pi'].values
                lambda_v_u_r    = df_u_r['lambda_v'].values

                win_index_v_r   = df_v_r['win_index'].values
                eps_v_r         = df_v_r['eps'].values
                pi_v_r          = df_v_r['pi'].values
                lambda_u_v_r    = df_v_r['lambda_u'].values

                gamma_r_u_r     = np.where(win_index_u_r == 1, np.exp(eps_u_r), np.exp(-eps_u_r))
                gamma_r_v_r     = np.where(win_index_v_r == 1, np.exp(eps_v_r), np.exp(-eps_v_r))

                numerator       = 1 + np.sum(pi_u_r) + np.sum(1 - pi_v_r)
                denominator     = 2 / (1 + exp_scores[idx]) + \
                              np.sum(gamma_r_u_r / (gamma_r_u_r * exp_scores[idx] + lambda_v_u_r)) + \
                              np.sum(1 / (gamma_r_v_r * lambda_u_v_r + exp_scores[idx]))

                exp_scores[idx] = numerator / denominator
            converged_i, _   = inner_checker.update(exp_scores,privileges,val_probs)
            if converged_i   == 0: break
        ################################################################################################################

        converged_o,_  = outer_checker.update(exp_scores,privileges,val_probs)

        current_val_probs   =   {type: val_probs[v]      for type, v in val_type_map.items()}
        current_privileges  =   {type:-privileges[v]     for type, v in priv_type_map.items()}
        logging.info(f'Reached iteration {_}. '
                     f'Current valence probabilities: {current_val_probs}. '
                     f'Current privileges: {current_privileges}. ')


        if converged_o == 0: break

    # convert scores, valence probabilities, privileges back to the original values
    ####################################################################################################################
    exp_scores  = {individual: exp_scores[v]     for individual, v  in indiv_map.items()}
    val_probs   = {type:       val_probs[v]      for type, v        in val_type_map.items()}
    privileges  = {type:      -privileges[v]     for type, v        in priv_type_map.items()}

    return exp_scores, val_probs, privileges