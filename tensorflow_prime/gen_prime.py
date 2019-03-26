def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n

def _not_divisible(n):
    return  lambda x:x%n>0

def primes():
    yield 2
    it=_odd_iter()
    while True:
        n=next(it)
        yield n
        it=filter(_not_divisible(n),it)

def to_bin(x,bins=20):
    str=bin(x)[2:].zfill(bins)
    return [int(b) for b in str]

import time
start_time = time.time()
lens=500000
with open('prime.txt','w') as f:
    for n in primes():
        if n<lens:
           # write to file
            f.write(str(n)+ '\n')
        else:
            break
    f.close()
print('finished... time: ',time.time() - start_time)