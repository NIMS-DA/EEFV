import numpy as np
import itertools
import csv
import physbo

##################
# parameters
##################

num_try = 100
num_initial = 10

##################
# training data
##################

csv_file = open("./data.csv", "r")

h = next(csv.reader(csv_file))

x_compositions = []
t1 = []

for row in csv.reader(csv_file):
    x_compositions.append(row[1])
    t1.append(float(row[2]))


class simulator:

    def __call__(self, action):
        action_idx = action[0]
        fx = t1[action_idx]

        return - fx


x_elements = []

for i in range(len(x_compositions)):

    x_elements.append([x_compositions[i][3:5], x_compositions[i][5:7], x_compositions[i][7:9]])


x_elements_all = itertools.chain.from_iterable(x_elements)

elements = list(set(itertools.chain.from_iterable(x_elements)))
elements.sort()

X = []

for i in range(len(x_elements)):

    each_bits = []

    for j in range(len(elements)):

        if elements[j] in x_elements[i]:
            each_bits.append(1)
        else:
            each_bits.append(0)

    X.append(each_bits)


max_step_res_bo = []
time_profile_bo = []
time_profile_max_bo = []


max_step_res_ra = []
time_profile_ra = []
time_profile_max_ra = []


for j in range(num_try):

    # BO
    
    policy = physbo.search.discrete.policy(test_X = np.array(X))
    policy.set_seed(j + 2222)

    res = policy.random_search(max_num_probes=num_initial, simulator=simulator())


    res = policy.bayes_search(max_num_probes=len(X)-num_initial, simulator=simulator(), score='TS',interval=5, num_rand_basis=100)

    results = res.fx[0:res.total_num_search]

    for i in range(len(results)):
        if results[i] == max(results):
            maxstep = i

    max_profile = []

    max_value = results[0]

    for i in range(len(results)-1):

        if max_value < results[i + 1]:

            max_value = results[i + 1]

            max_profile.append(max_value)

        else:

            max_profile.append(max_value)


    max_step_res_bo.append(maxstep)
    time_profile_bo.append(results)
    time_profile_max_bo.append(max_profile)



    # Random
    
    policy = physbo.search.discrete.policy(test_X = np.array(X))
    policy.set_seed(j + 2222)

    res = policy.random_search(max_num_probes=len(X), simulator=simulator())

    results = res.fx[0:res.total_num_search]

    for i in range(len(results)):
        if results[i] == max(results):
            maxstep = i

    max_profile = []

    max_value = results[0]

    for i in range(len(results)-1):

        if max_value < results[i + 1]:

            max_value = results[i + 1]

            max_profile.append(max_value)

        else:

            max_profile.append(max_value)
    
    max_step_res_ra.append(maxstep)
    time_profile_ra.append(results)
    time_profile_max_ra.append(max_profile)


print("BO max step mean = ", np.mean(max_step_res_bo))
print("BO max step std = ", np.std(max_step_res_bo))

print("")

print("Random max step mean = ", np.mean(max_step_res_ra))
print("Random max step std = ", np.std(max_step_res_ra))


