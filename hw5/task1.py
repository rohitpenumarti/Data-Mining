import sys
from random import randint
import binascii
from blackbox import BlackBox
from math import log
from time import perf_counter

def init_global_vars(stream_size, num_asks):
    global LEN_BIT_ARRAY
    global NUM_HASH_FUNCS
    global HASH_FUNCS

    LEN_BIT_ARRAY = 69997
    NUM_HASH_FUNCS = determine_num_hash(stream_size*num_asks, LEN_BIT_ARRAY)
    HASH_FUNCS = create_hash_functions(NUM_HASH_FUNCS)

def determine_num_hash(m, n):
    return round((n/m)*log(2))

def hash_func(a, b, p):
    return lambda x: ((a*x+b)%p)%LEN_BIT_ARRAY

def create_hash_functions(num_hash_functions):
    p = 4294967311
    return [hash_func(randint(1, 2**32-1), randint(0, 2**32-1), p) for _ in range(num_hash_functions)]

def myhashs(s):
    result = []
    for f in HASH_FUNCS:
        result.append(f(s))
    return result

class BloomFilter():

    def initalize(self):
        self.bit_array = [0]*LEN_BIT_ARRAY

    def bloom_filter(self, stream_users):
        bloom_seen_user_set = set([])
        true_seen_user_set = set([])
        stream_users_int = list(map(lambda x: int(binascii.hexlify(x.encode('utf8')),16), stream_users))
        for user_int in stream_users_int:
            user_hashes = myhashs(user_int)
            count = 0
            test_bit_array = [0]*LEN_BIT_ARRAY
            for user_hash in user_hashes:
                if self.bit_array[user_hash] == 1:
                    count += 1
                else:
                    test_bit_array[user_hash] = 1
            
            if count != len(user_hashes):
                self.bit_array = [ba+tba for ba, tba in zip(self.bit_array, test_bit_array)]
                bloom_seen_user_set.add(user_int)
            true_seen_user_set.add(user_int)
        return bloom_seen_user_set, true_seen_user_set

    def compute_fpr(self, bloom_seen_user_set, true_seen_user_set):
        num_false_seen = len(true_seen_user_set-bloom_seen_user_set)
        return num_false_seen/len(true_seen_user_set)

if __name__ == "__main__":
    start = perf_counter()
    input_file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]

    bx = BlackBox()
    bf = BloomFilter()
    res = []
    init_global_vars(stream_size, num_of_asks)
    bf.initalize()
    for i in range(num_of_asks):
        stream_users = bx.ask(input_file_name, stream_size)
        bloom_seen_user_set, true_seen_user_set = bf.bloom_filter(stream_users)
        fpr = bf.compute_fpr(bloom_seen_user_set, true_seen_user_set)
        res.append((i, fpr))

    with open(output_file_name, 'w') as file:
        file.write('Time,FPR\n')
        for i, pair in enumerate(res):
            if i == len(res)-1:
                file.write(f'{pair[0]},{pair[1]}')
            else:
                file.write(f'{pair[0]},{pair[1]}\n')

    end = perf_counter()
    print(f'Time elapsed: {end-start}')