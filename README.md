# Parallel Quicksort

## Programming assignment 2 for CSE-6220

This program implements parallel quicksort using MPI in C++17.

## Make and run

Run the following to make and run the program `pqsort` with input file `pa2_input.txt` and output file `output.txt`, using 8 processors

```shell
$ make run in=pa2_input.txt out=output.txt p=8
```

By default, `p=4`.
