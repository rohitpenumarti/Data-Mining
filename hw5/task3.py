import sys
from random import randint, seed, random
from blackbox import BlackBox
from time import perf_counter

SEQ_NUM = 100

def resevoir_sampling(curr_sample, curr_res, stream_users, i):
    sample = curr_sample
    res = curr_res
    if SEQ_NUM*i == 100:
        sample = stream_users
    else:
        for j, user in enumerate(stream_users):
            prob = random()
            if prob < 100/(SEQ_NUM*(i-1)+j+1):
                idx = randint(0, 99)
                sample[idx] = user
    
    res.append([SEQ_NUM*i, sample[0], sample[20], sample[40], sample[60], sample[80]])
    return sample, res

if __name__ == "__main__":
    start = perf_counter()
    input_file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]

    seed(553)
    bx = BlackBox()
    
    res = []
    sample = []
    for i in range(num_of_asks):
        stream_users = bx.ask(input_file_name, stream_size)
        sample, res = resevoir_sampling(sample, res, stream_users, i+1)

    with open(output_file_name, 'w') as file:
        file.write('seqnum,0_id,20_id,40_id,60_id,80_id\n')
        for i, seq in enumerate(res):
            if i == len(res)-1:
                file.write(f'{seq[0]},{seq[1]},{seq[2]},{seq[3]},{seq[4]},{seq[5]}')
            else:
                file.write(f'{seq[0]},{seq[1]},{seq[2]},{seq[3]},{seq[4]},{seq[5]}\n')

    end = perf_counter()
    print(f'Time elapsed: {end-start}')