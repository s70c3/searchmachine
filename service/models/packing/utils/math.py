def mean(arr):
    return sum(arr) / len(arr)

def round_arr(arr, digits=2):
    return list(map(lambda n: round(n, digits), arr))

def mul(arr):
    if len(arr) > 1:
        return arr[0] * mul(arr[1:]) 
    else:
        return arr[0]