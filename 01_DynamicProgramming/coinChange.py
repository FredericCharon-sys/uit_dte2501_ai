def coin_change(sum):
    """
    This is the coin_change method, it takes a sum for an input and returns the minimum number of coins you need to
    change this sum.
    """

    """
    Creating a new table with only one entry for the sum of 0 and initializing an integer for each coin
    """
    coin_change_dp = [0]
    nok20 = 20
    nok10 = 10
    nok5 = 5
    nok1 = 1

    """
    this for-statement is adding a new entry to the table for each value in between 0 and the sum
    """
    for i in range(1, sum + 1):

        """
        The following if statements make sure that we don't try to use coins with a higher number than our value.
        In each iteration we will append one item to our table using the Tabular Coin Change method from the lecture.
        """
        if i < nok20:
            if i < nok10:
                if i < nok5:
                    coin_change_dp.append(coin_change_dp[i - nok1] + 1)
                else:
                    coin_change_dp.append(min(coin_change_dp[i - nok1], coin_change_dp[i - nok5]) + 1)
            else:
                coin_change_dp.append(min(coin_change_dp[i - nok1], coin_change_dp[i - nok5],
                                          coin_change_dp[i - nok10]) + 1)
        else:
            coin_change_dp.append(min(coin_change_dp[i - nok1], coin_change_dp[i - nok5],
                                      coin_change_dp[i - nok10], coin_change_dp[i - nok20]) + 1)

    '''
     finally we print out our result
    '''
    result = coin_change_dp[sum]
    print("To change the sum of %d NOK you need a minimum of %d coin(s)." % (sum, result))


coin_change(1040528)
