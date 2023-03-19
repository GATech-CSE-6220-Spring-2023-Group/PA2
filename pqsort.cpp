#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <string>
#include <vector>

using std::string;
using std::to_string;
using std::vector;
using std::cout;

constexpr int ROOT = 0;

void quicksort_serial(vector<int>& values, int left, int right) {
    if (left >= right || values.empty()) return;

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
  Communicator `comm` has `q` processors on which `m` block-distributed integers should be sorted.
  `values_local` contains the contiguous subset of the `m` unsorted values distributed to this processor.

  If `q = 1`, sort the values serially on the group's only processor.
  Otherwise, sort in parallel as follows:
  * All processors in `comm` generate a random number `k` between 0 and m − 1.
  * The processor that has the pivot broadcasts it to all processors in the communicator.
  * Each processor partitions its values into two subarrays containing:
    1) `values_l_local`: values <= pivot, of size `m_l_local`, and
    2) `values_r_local`: values > pivot, of size `m_r_local`.
  * Use an `AllGather` to gather the `m_l_local` and `m_r_local` subarray sizes for each processor across every processor in `comm`,
    storing in arrays `M_l` and `M_r` respectively.
  * Using `M_l` and `M_r`, compute the total number of integers less than or equal to the pivot (`m_l`) and greater than the pivot (`m_r`).
  * Partition the `q` processors into two subproblems of sorting `m_l` and `m_r` values by allocating processors proportionally.
  * On each processor, use `M_l` and `M_r` to compute where to send its data and from where to receive its data.
  * Create two new communicators corresponding to the two partitions, and use an `AllToAll` to perform the data transfer.
  * Recursively call `quicksort_parallel` within each partition.
*/
void quicksort_parallel(vector<int> &values_local, size_t m, MPI_Comm comm = MPI_COMM_WORLD) {
    int q, r; // Communicator size and rank.
    MPI_Comm_size(comm, &q);
    MPI_Comm_rank(comm, &r);

    const int m_local = values_local.size();
    if (m_local == 0) return; // Nothing to do.
    if (q == 0) throw std::runtime_error("quicksort_parallel: Empty communicator assigned " + to_string(m_local) + " values to sort.");
    if (q == 1) return quicksort_serial(values_local, 0, m_local - 1); // Only one processor in `comm`. Sort serially.

    // Use the same random seed on all processors within `comm`, so that each processor generates the same random number.
    static const int seed = 123;
    std::mt19937 rand_gen(seed);
    std::uniform_int_distribution<> rand_dist(0, m - 1);
    const int pivot_i = rand_dist(rand_gen);
    const int pivot_r = pivot_i / m_local; // Rank of processor that has the pivot.

    int pivot;
    if (r == pivot_r) {
        const int local_i = (m / q) * r; // Index of first value on this processor (where `local_values` starts in the comm-wide value array).
        pivot = values_local[pivot_i - local_i];
    }
    MPI_Bcast(&pivot, 1, MPI_INT, pivot_r, MPI_COMM_WORLD);

    int m_l_local = 0, m_r_local = 0;
    for (int i = 0; i < values_local.size(); i++) {
        if (values_local[i] <= pivot) m_l_local++;
        else m_r_local++;
    }

    // Gather `m_l_local` and `m_r_local` from all processors.
    vector<int> M_l(q), M_r(q);
    MPI_Allgather(&m_l_local, 1, MPI_INT, &M_l[0], 1, MPI_INT, comm);
    MPI_Allgather(&m_r_local, 1, MPI_INT, &M_r[0], 1, MPI_INT, comm);

    //  Using `M_l` and `M_r`, compute the total number of integers less than or equal to the pivot (`m_l`) and greater than the pivot (`m_r`).
    int m_l = 0, m_r = 0;
    for (int i = 0; i < q; i++) {
        m_l += M_l[i];
        m_r += M_r[i];
    }

    // Partition the `q` processors into two subproblems of sorting `m_l` and `m_r` values by allocating processors proportionally...

    // Compute the number of processors in each group by satisfying the following constraints:
    // * `q_l + q_r = q` (use all processors)
    // * `q_l/q_r = m_l/m_r` (assign processors proportionally to problem sizes)
    // * `q_l > 0, q_r > 0` (assign at least one processor to each group)
    const int q_l = std::max(1, int(round(float(q * m_l) / float(m_l + m_r)))); // Round to nearest integer instead of truncating & prevent 0 size.
    const int q_r = q - q_l;
    const bool is_left = r < q_l;

    // Compute where to send and from where to receive data.
    vector<int> send_counts(q), recv_counts(q);
    vector<int> send_displs(q), recv_displs(q);
    for (int i = 0; i < q; i++) {
        // TODO handle rounding
        send_counts[i] = M_l[i] + M_r[i];
        recv_counts[i] = i < q_l ? m_l / q_l : m_r / q_r;
        send_displs[i] = i == 0 ? 0 : send_displs[i - 1] + send_counts[i - 1];
        recv_displs[i] = i == 0 ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
    }

    // Create two new communicators corresponding to the two partitions.
    MPI_Comm new_comm;
    MPI_Comm_split(comm, is_left, r, &new_comm);

    cout << "Processor " << r << '\n';
    cout << "\tpivot " << pivot << '\n';
    cout << "\tm_l = " << m_l << ", m_r = " << m_r << '\n';
    cout << "\tm_l_local = " << m_l_local << ", m_r_local = " << m_r_local << '\n';
    cout << "\tq_l = " << q_l << ", q_r = " << q_r << '\n';
    cout << "\tgroup = " << (is_left ? "left" : "right") << '\n';
    cout << "\tsend_counts = " << send_counts[r] << ", recv_counts = " << recv_counts[r] << '\n';
    cout << "\tsend_displs = " << send_displs[r] << ", recv_displs = " << recv_displs[r] << '\n';

    // Use an `AllToAll` to perform the data transfer.
    vector<int> recv_values(recv_counts[r]);
    MPI_Alltoallv(&values_local[0], &send_counts[0], &send_displs[0], MPI_INT, &recv_values[0], &recv_counts[0], &recv_displs[0], MPI_INT, new_comm);

    // Recursively sort the two partitions.
    quicksort_parallel(recv_values, is_left ? m_l : m_r, new_comm);

    MPI_Comm_free(&new_comm);
}

int main(int argc, char* argv[]) {
    if (argc < 3) throw std::invalid_argument("Usage: `pqsort {input_file} {output_file}`");

    MPI_Init(nullptr, nullptr);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    const bool is_root = world_rank == ROOT;

    int n; // Number of integers to sort.
    vector<int> values_global;

    if (is_root) {
        const string input_file_path = argv[1];
        std::ifstream input_file_stream(input_file_path);
        if (!input_file_stream.is_open()) throw std::invalid_argument("Failed to open input file with path: " + input_file_path);

        // Assume the input file is formatted as follows:
        // 1. The first line contains the number of values to sort, `n`.
        // 2. The second line contains at least `n` values to sort, separated by spaces.
        input_file_stream >> n;

        values_global.resize(n);
        for (int i = 0; i < n; i++) input_file_stream >> values_global[i];
        input_file_stream.close();
    }

    MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    const int n_local = n / world_size; // Number of values to sort on this process. TODO handle `n % world_size != 0`.
    vector<int> values_local(n_local); // Numbers to sort on this process.

    // Block-distribute the array to all processors
    MPI_Scatter(&values_global[0], n_local, MPI_INT, &values_local[0], n_local, MPI_INT, ROOT, MPI_COMM_WORLD);

    const double start_time_s = MPI_Wtime();
    // For now, pretend we have a communication group of size 1 on each processor,
    // and just sort each processor's local numbers serially.
    quicksort_parallel(values_local, n, MPI_COMM_WORLD);

    // Send values to root, overwriting the `values_global` vector.
    MPI_Gather(&values_local[0], n_local, MPI_INT,  &values_global[0], n_local, MPI_INT, ROOT, MPI_COMM_WORLD);
    const double end_time_s = MPI_Wtime();

    if (is_root) {
        const string output_file_path = argv[2];
        std::ofstream output_file_stream(output_file_path);
        if (!output_file_stream.is_open()) throw std::invalid_argument("Failed to open output file with path: " + output_file_path);

        for (int i = 0; i < n; i++) {
            output_file_stream << values_global[i] << (i == n - 1 ? "" : " ");
        }
        output_file_stream << '\n' << std::fixed << std::setprecision(6) << end_time_s - start_time_s << '\n';
        output_file_stream.close();
    }
    MPI_Finalize();

    return 0;
}
