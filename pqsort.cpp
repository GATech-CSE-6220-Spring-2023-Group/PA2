#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

using std::string;
using std::vector;

constexpr int ROOT = 0;

void quicksort_serial(vector<int>& values, int left, int right) {
    if (left >= right) return;

    const int pivot = values[left + (right - left) / 2];
    int i = left, j = right;
    while (i <= j) {
        while (values[i] < pivot) i++;
        while (values[j] > pivot) j--;
        if (i <= j) std::swap(values[i++], values[j--]);
    }
    quicksort_serial(values, left, j);
    quicksort_serial(values, i, right);
}

/**
  If `q = 1`, sort the values serially on group's only processor.
  Otherwise, sort in parallel as follows:
  * All processors in `comm` generate a random number `k` between 0 and m − 1.
  * The processor that has the kth integer (the pivot) broadcasts it to all processors in the communicator.
  * Each processor partitions its integers values into two subarrays containing:
    1) `values_l`: values <= pivot, of size `m_l`, and
    2) `values_r`: values > pivot, of size `m_r`.
  * Using an `AllGather`, gather the `m_l` and `m_r` subarray sizes for each processor across every processor in `comm`,
    storing in arrays `M_l` and `M_r` respectively.
  * Partition the `q` processors into two subproblems of sorting `m_l` values and `m_r` values by allocating processors in proportion to `m_l` and `m_r`.
  * On each processor, use `M_l` and `M_r` to compute where to send its data and from where to receive its data.
  * Create two new communicators corresponding to the two partitions, and use an `AllToAll` to perform the data transfer.
  * Recursively call `quicksort_parallel` within each partition.
*/
void quicksort_parallel(vector<int> &values, size_t m, int q, MPI_Comm comm = MPI_COMM_WORLD) {
    if (q == 1) {
        quicksort_serial(values, 0, m - 1);
        return;
    }
    // TODO parallel quicksort
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
    // For now, pretend we have a communication group of size 1 on each processor,
    // and just sort each processor's local numbers serially.
    quicksort_parallel(local_numbers, local_numbers.size(), 1, MPI_COMM_WORLD);

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
