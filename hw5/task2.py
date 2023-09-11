import sys
from random import randint
import binascii
from blackbox import BlackBox
from math import pow

def init_global_vars():
    global NUM_HASH_FUNCS
    global NUM_BINS
    global HASH_FUNCS

    NUM_HASH_FUNCS = 100
    NUM_BINS = 300
    HASH_FUNCS = create_hash_functions(NUM_HASH_FUNCS, NUM_BINS)

def hash_func(a, b, p, m):
    return lambda x: ((a*x+b)%p)%m

def create_hash_functions(num_hash_functions, num_bins):
    p = 4294967311
    return [hash_func(randint(1, 2**32-1), randint(0, 2**32-1), p, num_bins) for _ in range(num_hash_functions)]

def myhashs(s):
    result = []
    for f in HASH_FUNCS:
        result.append(f(s))
    return result

def flajolet_martin(stream_users, num_splits):
    stream_users_int = list(map(lambda x: int(binascii.hexlify(x.encode('utf8')),16), stream_users))
    num_spots = len(f'{NUM_BINS}:0b')
    all_user_bins = []
    for user_int in stream_users_int:
        user_hashes = myhashs(user_int)
        all_user_bins.append([len(f'{user_hash:0{num_spots}b}') - len(f'{user_hash:0{num_spots}b}'.rstrip('0')) \
            for user_hash in user_hashes])
    
    transpose_user_bins = list(map(list, zip(*all_user_bins)))
    estimates = [pow(2, max(len_trailing_zeros)) for len_trailing_zeros in transpose_user_bins]
    partition_length = int(len(estimates)/num_splits)
    partitioned_estimates = [estimates[i:i+partition_length] for i in range(0, len(estimates), partition_length)]
    average_partitioned_estimates = sorted([sum(partition)/partition_length for partition in partitioned_estimates])

    if num_splits%2 == 0:
        return (average_partitioned_estimates[num_splits/2-1]+average_partitioned_estimates[num_splits/2])/2
    else:
        return average_partitioned_estimates[int(num_splits/2)]

if __name__ == "__main__":
    input_file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]

    bx = BlackBox()
    stream_users = bx.ask(input_file_name, stream_size)
    res = []
    init_global_vars()
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file_name, stream_size)
        num_unique_users = len(set(stream_users))
        estimate = flajolet_martin(stream_users, 5)
        res.append((num_unique_users, estimate))

    with open(output_file_name, 'w') as file:
        file.write('Time,Ground Truth,Estimation\n')
        for i, pair in enumerate(res):
            if i == len(res)-1:
                file.write(f'{i},{pair[0]},{int(pair[1])}')
            else:
                file.write(f'{i},{pair[0]},{int(pair[1])}\n')