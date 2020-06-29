# Rivki Kanterovich
# ID: 212030761
# Geula Shoshan
# ID: 11826658
import math
import csv
import bcolors
from collections import Counter
from pathlib import Path
from prettytable import PrettyTable

bcolors = bcolors.bcolors()

#Class like enum of the heuristics distance avilable
class Heuristic:
    EUCLIDEAN = 0
    MANHATTAN = 1
    HAMMING = 2

#class with ctor that has distance and tag properties
class distClass:
    def __init__(self, dist=-1, tag='-'):
        self.dist = dist  # distance of current point from test point
        self.tag = tag  # tag of current point


#calculate the distance between 2 vectors by selected heuristic
def calculate_distance(instance1, instance2, heuristic: Heuristic):
    if (heuristic == Heuristic.EUCLIDEAN):
        return euclidean_distance(instance1, instance2)
    if (heuristic == Heuristic.MANHATTAN):
        return manhattan_distance(instance1, instance2)
    if (heuristic == Heuristic.HAMMING):
        return hamming_distance(instance1, instance2)

# check the euclidean distances between 2 vectors
def euclidean_distance(instance1, instance2):
    distance = 0
    length = len(instance1)
    for x in range(length - 1):
        # print ('x is ' , x)
        num1 = float(instance1[x])
        num2 = float(instance2[x])
        distance += pow(num1 - num2, 2)
    return math.sqrt(distance)


# Manhattan distance between vectors
def manhattan_distance(instance1, instance2):
    n = len(instance1)-1
    sum = 0
    # for each point, finding distance
    # to rest of the point
    for i in range(n):
        sum += abs(float(instance1[i]) - float(instance2[i]))
    return sum


# hamming distance between vectors
def hamming_distance(instance1, instance2):
    n = len(instance1)
    return sum(c1 != c2 for c1, c2 in zip(instance1[:n-1], instance2[:n-1]))

# print just the vectors
def print_vector(vector):
    # add comma to the list
    print('[', ','.join(map(str, vector[:-1])), ']', end="")

# print the vector with his tag
def print_vector_with_tag(vector):
    length = len(vector)
    print('The vector', end=" ")
    print_vector(vector)
    print(' has tag', get_vector_tag(vector))


# get the vector original tag
def get_vector_tag(vector):
    return vector[len(vector) - 1]


# print distance between 2 vectors
def print_distance(vector1, vector2, heuristic: Heuristic):
    print("\tThe distance between: ", end="")
    print_vector(vector1)
    print(" and ", end="")
    print_vector(vector2)
    print(" is: ", end="")
    print(calculate_distance(vector1, vector2, heuristic))


# read the csv file and return the dataset
def read_file(path):
    with open(path, 'r') as myCsvfile:
        lines = csv.reader(myCsvfile)
        dataWithHeader = list(lines)
        # put data in dataset without header line
        headers = dataWithHeader[:1]  # return array of the columns names
        ds = dataWithHeader[1:]
        return ds, headers[0]


# read the csv file and return the dataset
def write_file(path, dataWithHeader):
    with open(path, 'w', newline='') as myCSVtest:
        writer = csv.writer(myCSVtest)
        writer.writerows(dataWithHeader)


# calculate the distance between vector and datasets
def calculate_distance_vector_and_dataset(vector, ds, heuristic: Heuristic):
    eucDistances = []  # list of distances, will hold objects of type distClass
    for i in range(len(ds)):
        d = (calculate_distance(vector, ds[i], heuristic))
        eucDistances.append(distClass(d, get_vector_tag(ds[i])))  # one record's distance and tag
    eucDistances.sort(key=lambda x: x.dist)
    return eucDistances

# calculate the distance between vector and datasets with printing options
def calc_distance_vector_from_ds(vector, ds, output_print, heuristic: Heuristic):
    if (output_print):
        print("The vector:", end="")
        print_vector(vector)
        print("")
    distance_ds = calculate_distance_vector_and_dataset(vector, ds, heuristic)
    if (output_print):
        t = PrettyTable()
        t.field_names = ["distance", "tag"]
        for i in range(len(distance_ds)):
            t.add_row([distance_ds[i].dist, distance_ds[i].tag])
        print(t)
    return distance_ds

#get the top 1 tag from a sorted dataset
def calc_the_tag_value(distance_sorted_ds, k_value):
    # find the most tag in the table that sorted already  and return the best tag
    counts = Counter([o.tag for o in distance_sorted_ds[:k_value]])
    # return just the top value
    return counts.most_common(1)[0][0]

"""
main function that getting the original dataset and tester dataset k value and heuristic
and return the tester_ds data with accurance (ממוצע של תוצאה נכונה )
"""
def  calc_calculated_tags_from_datasets(original_ds, tester_ds, headers, k_value, output_print, heuristic: Heuristic):
    true_founded = 0
    # wrond_founded =0
    accuracy=0;
    rows = len(tester_ds)
    headers.append("K=" + str(k_value))
    for i in range(rows):
        # find all the distance between the vector and the dataset
        distance_sorted = calc_distance_vector_from_ds(tester_ds[i], original_ds, output_print, heuristic)
        # the calculation tag
        original_tag = get_vector_tag(tester_ds[i])
        founded_tag = calc_the_tag_value(distance_sorted, k_value)
        if(founded_tag == original_tag):
            true_founded+=1
        # else
        #     wrond_founded+=1
        tester_ds[i].append(founded_tag)
    tester_ds.insert(0, headers)
    accuracy = true_founded / rows
    return tester_ds, accuracy

"""
this function based on calc_calculated_tags_from_datasets and write the result to csv file
"""
def write_kx_to_csv(original_file_path, tester_file_path, k_value, output_print, heuristic: Heuristic, out_file_name):
    original_ds, original_headers = read_file(original_file_path)
    ds_test, tester_headers = read_file(tester_file_path)
    tester_with_calculation_tag, accuracy = calc_calculated_tags_from_datasets(original_ds, ds_test, tester_headers, k_value,
                                                                     output_print, heuristic)
    write_file(out_file_name, tester_with_calculation_tag)
    return accuracy


"""
helper function for run on 3 k value for selected heuristic
"""
def helper_write_multi_ks(heuristic: Heuristic,  heuristic_name,  first_letter):
    best_accuracy=1
    accuracy1 = write_kx_to_csv('mytrain.csv', 'mytest.csv', 1, 0, heuristic, 'csv/mytest1' + first_letter + '.csv')
    print("\taccuracy of  train using "  + heuristic_name + " k=1 ", accuracy1)
    accuracy7 = write_kx_to_csv('mytrain.csv', 'mytest.csv', 7, 0, heuristic, 'csv/mytest7' + first_letter + '.csv')
    if(accuracy7 > accuracy1) : best_accuracy = 7
    print("\taccuracy of  train using "  + heuristic_name + " k=7 ", accuracy7)
    accuracy19 = write_kx_to_csv('mytrain.csv', 'mytest.csv', 19, 0, heuristic, 'csv/mytest19' + first_letter + '.csv')
    if (accuracy19 > accuracy7 and accuracy19 > accuracy1): best_accuracy = 19
    print("\taccuracy of  train using "  + heuristic_name + " k=19 ", accuracy19)
    print("\tThe best accuracy in heuristic: "  + heuristic_name + " is  k=", best_accuracy)



def main():
    Path("csv").mkdir(parents=True, exist_ok=True)
    print(bcolors.RED + bcolors.BOLD + "4.1.1" + bcolors.ENDC + ":")
    vectors = [[1, 0, 0, '?'], [1, 1, 1, 'M'], [1, 2, 0, 'F']]
    print("\tDONE")
    print(bcolors.RED + bcolors.BOLD + "4.1.2" + bcolors.ENDC + ":")
    for i in range(len(vectors)):
        print('\t', end="")
        print_vector_with_tag(vectors[i])
    print(bcolors.RED + bcolors.BOLD + "4.1.3" + bcolors.ENDC + ":")
    print_distance(vectors[1], vectors[2], Heuristic.EUCLIDEAN)
    print(bcolors.RED + bcolors.BOLD + "4.1.4" + bcolors.ENDC + ":")
    ds1, ds1_headers = read_file('myFile.csv')
    print_distance(ds1[0], ds1[1], Heuristic.EUCLIDEAN)
    print(bcolors.RED + bcolors.BOLD + "4.1.5" + bcolors.ENDC + ":")
    print("\tDONE")
    print(bcolors.RED + bcolors.BOLD + "4.1.6" + bcolors.ENDC + ":")
    print("\tDONE")
    print(bcolors.RED + bcolors.BOLD + "4.1.7" + bcolors.ENDC + ":")
    distances = calc_distance_vector_from_ds(ds1[0], ds1[1:], 1, Heuristic.EUCLIDEAN)
    print(bcolors.RED + bcolors.BOLD + "4.1.8" + bcolors.ENDC + ":")
    print("\tThe most tag with k=1 is:", calc_the_tag_value(distances, 1))
    print(bcolors.RED + bcolors.BOLD + "4.1.9" + bcolors.ENDC + ":")
    print("\tThe most tag with k=3 is:", calc_the_tag_value(distances, 3))
    print(bcolors.BLUE + bcolors.BOLD + "4.2.1" + bcolors.ENDC + ":")
    accuracy = write_kx_to_csv('myFile.csv', 'myFile_test.csv', 3, 1, Heuristic.EUCLIDEAN, 'csv/myFile_test3e.csv')
    print("\taccuracy of test using  EUCLIDEAN k=3 ", accuracy)
    print(bcolors.BLUE + bcolors.BOLD + "4.2.2.a" + bcolors.ENDC + ":")
    helper_write_multi_ks(Heuristic.EUCLIDEAN, "EUCLIDEAN", "e")
    print(bcolors.BLUE + bcolors.BOLD + "4.2.2.e" + bcolors.ENDC + ":")
    helper_write_multi_ks(Heuristic.MANHATTAN, "MANHATTAN", "m")
    print(bcolors.BLUE + bcolors.BOLD + "4.2.2.f" + bcolors.ENDC + ":")
    helper_write_multi_ks(Heuristic.HAMMING, "HAMMING", "h")


main()
