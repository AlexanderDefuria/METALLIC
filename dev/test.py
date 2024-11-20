from pathos.pools import ProcessPool
import sys
import time


def test(i):
    i = 1
    start = time.time()
    while True:
        i *= 3 
        i = i % 21311
        

if __name__ == "__main__":
   with ProcessPool() as p:
        for i, result in enumerate(p.imap(test, [1] * 10)):
            print(i, result)
