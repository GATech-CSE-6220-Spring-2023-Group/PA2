#include <fstream>
#include <mpi.h>
#include <string>

using std::string;

int main(int argc, char* argv[]) {
    if (argc < 3) throw std::invalid_argument("Usage: `pqsort {input_file} {output_file}`");

    const string input_file_path = argv[1];
    std::ifstream input_file_stream(input_file_path);
    if (!input_file_stream.is_open()) throw std::invalid_argument("Failed to open input file with path: " + input_file_path);

    // Assume the input file is formatted as follows:
    // 1. The first line contains the number `n` of integers to sort.
    // 2. The second line contains at least `n` integers to sort, separated by spaces.

    int n;
    input_file_stream >> n;

    int unsorted_numbers[n];
    for (int i = 0; i < n; i++) input_file_stream >> unsorted_numbers[i];
    input_file_stream.close();

    MPI_Init(nullptr, nullptr);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int sorted_numbers[n];
    const double start_time_s = MPI_Wtime();
    // For now, just copy the unsorted numbers to the sorted numbers.
    for (int i = 0; i < n; i++) {
        sorted_numbers[i] = unsorted_numbers[i];
    }
    const double end_time_s = MPI_Wtime();

    const string output_file_path = argv[2];
    std::ofstream output_file_stream(output_file_path);
    if (!output_file_stream.is_open()) throw std::invalid_argument("Failed to open output file with path: " + output_file_path);

    for (int i = 0; i < n; i++) {
        output_file_stream << sorted_numbers[i] << (i == n - 1 ? "" : " ");
    }
    output_file_stream << '\n' << std::fixed << std::setprecision(6) << end_time_s - start_time_s << '\n';
    output_file_stream.close();

    MPI_Finalize();

    return 0;
}
