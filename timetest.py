import time

tic = time.perf_counter()

for i in range(int(800*4)):
    a = time.perf_counter()
    b = time.perf_counter() - a

toc = time.perf_counter() - tic
print(toc)

