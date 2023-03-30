import random
import argparse

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Generate a file with a list of random integers.")
parser.add_argument("n", type=int, help="Number of integers in the list.")
parser.add_argument("filepath", type=str, help="File path.")
args = parser.parse_args()

# Get the desired number of integers from the command line argument
n = args.n
filepath = args.filepath

# Generate a list of random integers of length n
rand_integers = [random.randint(1, 100) for _ in range(n)]

# Write the two-line file
with open(filepath, "w") as f:
    f.write(f"{n}\n")
    f.write(" ".join(str(i) for i in rand_integers))

