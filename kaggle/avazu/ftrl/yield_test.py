# Auther        : wangrc
# Date          : 2019-01-10
# Description   :
# Refers        :
# Returns       :
import argparse
from pyspark import sql


# encoding:UTF-8
def yield_test(n):
    for i in range(2,n):
        yield n2(i)
        yield n3(i)
        yield n4(i)
        print("i=", i)
    print("do something.")
    print("end.")


def n2(i):
    return i ** 2

def n3(i):
    return i ** 3

def n4(i):
    return i ** 4


# print(2 ** 20)
for i in yield_test(10):
    print(i, "here")