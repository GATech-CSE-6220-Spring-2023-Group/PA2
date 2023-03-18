#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

using std::string;
using std::vector;

constexpr int ROOT = 0;

void quicksort(vector<int>& arr, int left, int right) {
    if (left >= right) return;

    const int pivot = arr[left + (right - left) / 2];
    int i = left, j = right;
    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) std::swap(arr[i++], arr[j--]);
    }
    quicksort(arr, left, j);
    quicksort(arr, i, right);
}

int main(int argc, char* argv[]) {
    if (argc < 3) throw std::invalid_argument("Usage: `pqsort {input_file} {output_file}`");

    MPI_Init(nullptr, nullptr);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const bool is_root = world_rank == ROOT;

    int n; // Number of integers to sort.
    vector<int> all_numbers;

    if (is_root) {
        const string input_file_path = argv[1];
        std::ifstream input_file_stream(input_file_path);
        if (!input_file_stream.is_open()) throw std::invalid_argument("Failed to open input file with path: " + input_file_path);

        // Assume the input file is formatted as follows:
        // 1. The first line contains the number `n` of integers to sort.
        // 2. The second line contains at least `n` integers to sort, separated by spaces.
        input_file_stream >> n;

        all_numbers.resize(n);
        for (int i = 0; i < n; i++) input_file_stream >> all_numbers[i];
        input_file_stream.close();
    }

    MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    const int n_local = n / world_size; // Number of integers to sort on this process. TODO handle `n % world_size != 0`.
    vector<int> local_numbers(n_local); // Numbers to sort on this process.

    // Block-distribute the array to all processors
    MPI_Scatter(&all_numbers[0], n_local, MPI_INT, &local_numbers[0], n_local, MPI_INT, ROOT, MPI_COMM_WORLD);

    const double start_time_s = MPI_Wtime();
    quicksort(local_numbers, 0, n_local - 1); // For now, each process sorts its own numbers.

    // Send numbers to root, overwriting the `all_numbers` vector.
    MPI_Gather(&local_numbers[0], n_local, MPI_INT,  &all_numbers[0], n_local, MPI_INT, ROOT, MPI_COMM_WORLD);
    const double end_time_s = MPI_Wtime();

    if (is_root) {
        const string output_file_path = argv[2];
        std::ofstream output_file_stream(output_file_path);
        if (!output_file_stream.is_open()) throw std::invalid_argument("Failed to open output file with path: " + output_file_path);

        for (int i = 0; i < n; i++) {
            output_file_stream << all_numbers[i] << (i == n - 1 ? "" : " ");
        }
        output_file_stream << '\n' << std::fixed << std::setprecision(6) << end_time_s - start_time_s << '\n';
        output_file_stream.close();
    }
    MPI_Finalize();

    return 0;
}
