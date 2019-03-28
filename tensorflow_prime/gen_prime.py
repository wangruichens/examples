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


def list_primes(n):
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if primes[i]:
            primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
    return primes

import time


# start_time = time.time()
# lens=10**1
# with open('prime_iter.txt','w') as f:
#     for n in primes():
#         if n<lens:
#            # write to file
#             f.write(str(n)+ '\n')
#         else:
#             break
#     f.close()
# elapse_slow=time.time() - start_time
# print('finished... time: ',elapse_slow)


# O(N*loglogN)
start_time = time.time()
lens=10**9
l=list_primes(lens)
with open('prime.txt','w') as f:
    for n in range(lens):
        if l[n]==True:
           # write to file
            f.write(str(n)+ '\n')
    f.close()
elapse_fast=time.time() - start_time
# 1 billion prime , cost 420s (7min)
print('finished ... time: ',elapse_fast)