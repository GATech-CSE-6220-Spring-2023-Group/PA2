# Argument defaults
p = 4
in = pa2_input.txt
out = output.txt

pqsort: pqsort.cpp
	mpicxx -std=c++17 -o pqsort pqsort.cpp

run: pqsort
	mpirun -np $(p) ./pqsort $(in) $(out)

clean:
	rm pqsort
