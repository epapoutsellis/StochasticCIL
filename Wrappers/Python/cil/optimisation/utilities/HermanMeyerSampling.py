import math
from itertools import islice, tee, chain
import logging

class HermanMeyerSampling:

    """
    Implementation of the Herman-Meyer sampling algorithm.

    Parameters
    ----------
    num_indices : int
        Total number of indices.
    num_batches : int, optional
        Number of batches. Defaults to None.
        If None, it is set to the same value as num_indices.

    Raises
    ------
    ValueError
        If num_batches or num_indices is a prime number.
    """    

    def __init__(self, num_indices, num_batches=None):
        
    
        self.num_indices = num_indices
        self.num_batches = num_batches
        
        if self.num_batches is None:
            self.num_batches = self.num_indices
        
        if (check_prime(self.num_batches) or check_prime(self.num_indices)):
            raise ValueError("Herman Meyer sampling requires a non-prime number for number of views and subsets.")            

        self.indices_used = []
        self.index = 0    

        if not self.num_indices%self.num_batches==0 :
            logging.warning("Batch size is not constant. ")                              

        # initial ordered list
        tmp_list = [i for i in range(self.num_indices)]  

        # split list into num_batches (iterators to list)
        splited_list_iter = tee(tmp_list, self.num_batches)    
        splited_list = [list(islice(it, index, None,  self.num_batches)) for index, it in enumerate(splited_list_iter)]        

        # define hm_order
        hm_order = herman_meyer_order(len(splited_list))

        # reorder list based on hm_order
        self.partition_list = [splited_list[i] for i in hm_order]


    def __next__(self):

        """
        Get the next partitioned list.

        Returns
        -------
        list
            Next partitioned list.

        """        

        if self.num_batches==self.num_indices:
            # case of only one item
            tmp_list = self.partition_list[self.index][0]
        else:
            tmp_list = self.partition_list[self.index]
            
        
        self.indices_used.append(tmp_list)         
        self.index+=1

        if self.index==len(self.partition_list):
            self.index=0            

        return tmp_list       


def prime_decomposition(n):

    """
    Perform prime decomposition of a given number.

    Parameters
    ----------
    n : int
        Number to decompose into prime factors.

    Returns
    -------
    list
        List of prime factors of the input number.

    Examples
    --------
    >>> prime_decomposition(12)
    [2, 2, 3]

    >>> prime_decomposition(29)
    [29]
    """    

    factors = []
    i = 2

    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)

    if n > 1:
        factors.append(n)

    return factors


def check_prime(n):

    """
    Check if a number is prime.

    Parameters
    ----------
    n : int
        Number to check for primality.

    Returns
    -------
    bool
        True if the number is prime, False otherwise.

    Examples
    --------
    >>> check_prime(17)
    True

    >>> check_prime(4)
    False
    """

    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def herman_meyer_order(n):

    """
    Generate the Herman-Meyer order for a given number.

    Parameters
    ----------
    n : int
        Number for which to generate the Herman-Meyer order.

    Returns
    -------
    list
        Herman-Meyer order for the given number.

    Examples
    --------
    >>> herman_meyer_order(15)
    [0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14]

    >>> herman_meyer_order(30)
    [0, 15, 5, 20, 10, 25, 1, 16, 6, 21, 11, 26, 2, 17, 7, 22, 12, 27, 3, 18, 8, 23, 13, 28, 4, 19, 9, 24, 14, 29]  

    >>> herman_meyer_order(7)
    [0, 1, 2, 3, 4, 5, 6]
    """    

    # prime decomposition
    factors = prime_decomposition(n)

    # len of prime factors
    n_factors = len(factors)

    # initialize
    order =  [0]*n

    value = 0

    for factor_n in range(n_factors):

        n_rep_value = 0

        if factor_n == 0:
            n_change_value = 1
        else:
            n_change_value = math.prod(factors[:factor_n])

        for element in range(n):

            mapping = value
            n_rep_value += 1
            if n_rep_value >= n_change_value:
                value = value + 1
                n_rep_value = 0
            if value == factors[factor_n]:
                value = 0
            order[element] = order[element] + math.prod(factors[factor_n+1:]) * mapping

    return order


