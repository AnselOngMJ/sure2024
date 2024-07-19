import itertools
import os

ARRAY_JOB_ID = 0
NUM_JOBS = 1407

failed = []
for i in range(NUM_JOBS):
    if str(i) not in os.listdir(f'./{ARRAY_JOB_ID}'):
        failed.append(i)
print(f'Number of jobs failed: {len(failed)}')

def get_ranges(i):
    for _, x in itertools.groupby(enumerate(i), lambda x: x[1] - x[0]):
        x = list(x)
        if x[0][1] == x[-1][1]:
            yield str(x[0][1])
        else:
            yield f'{x[0][1]}-{x[1][1]}'

print(','.join(list(get_ranges(failed))))
