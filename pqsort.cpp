#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numeric>
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

string to_string(vector<int> values) {
    string s = "[";
    for (int i = 0; i < values.size(); i++) {
        s += to_string(values[i]);
        if (i < values.size() - 1) s += ", ";
    }
    s += "]";
    return s;
}

 /**
  Communicator `comm` has `q` processors on which `m` block-distributed integers should be sorted.
  `values_local` contains the contiguous subset of the `m` unsorted values distributed to this processor.

  If `q = 1`, sort the values serially on the group's only processor.
  Otherwise, sort in parallel as follows:
  * All processors in `comm` generate a random number `k` between 0 and m − 1.
  * The processor that has the pivot broadcasts it to all processors in the communicator.
  * Each processor partitions its values into two subarrays, with left `m_l_local` values <= pivot and right `m_r_local` values > pivot.
  * Use an `AllGather` to gather the `m_l_local` and `m_r_local` subarray sizes for each processor across every processor in `comm`,
    storing in arrays `M_l` and `M_r` respectively.
  * Using `M_l` and `M_r`, compute the total number of integers <= pivot (`m_l`) and > pivot (`m_r`).
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

    const auto values_local_pre = values_local; // xxx making a copy only for debugging purposes.
    // Arrange local values so the left `m_l_local` values <= pivot and right `m_r_local` values > pivot.
    int i = 0, j = m_local - 1;
    while (i <= j) {
        while (i <= j && values_local[i] < pivot) i++;
        while (i <= j && values_local[j] > pivot) j--;
        if (i <= j) std::swap(values_local[i++], values_local[j--]);
    }
    const int m_l_local = i, m_r_local = m_local - i;

    // Gather `m_l_local` and `m_r_local` from all processors into `M_l` and `M_r`,
    // and compute the total number of integers <= pivot (`m_l`) and > pivot (`m_r`).
    vector<int> M_l(q), M_r(q);
    MPI_Allgather(&m_l_local, 1, MPI_INT, &M_l[0], 1, MPI_INT, comm);
    MPI_Allgather(&m_r_local, 1, MPI_INT, &M_r[0], 1, MPI_INT, comm);
    const int m_l = std::reduce(M_l.begin(), M_l.end());
    const int m_r = std::reduce(M_r.begin(), M_r.end());

    /* Partition the `q` processors into two subproblems of sorting `m_l` and `m_r` values by allocating processors proportionally. */

    /** 
      Compute the number of processors in the left/right groups by satisfying the following constraints:
      * `q_l + q_r = q` (use all processors)
      * `q_l/q_r = m_l/m_r` (assign processors proportionally to problem sizes)
      * `q_l > 0, q_r > 0` (assign at least one processor to each group)
    */
    int q_l = int(round(float(q * m_l) / float(m_l + m_r))); // Round to nearest integer instead of truncating.
    int q_r = q - q_l;
    if (q_l == 0) q_l = 1, q_r = q - 1;
    else if (q_r == 0) q_l = q - 1, q_r = 1;
    const bool is_left = r < q_l;

    // I found it difficult to calculate both the send and received counts, so at least for now,
    // I'm calculating the matrix of send counts for _all_ processors on this processor.
    // Then, I'm using this to find all `Alltoallv` values for this processor.
    int all_send_counts[q][q];
    for (int i = 0; i < q; i++) {
        const bool is_i_left = i < q_l; // Is processor `i` in the left group?
        // Distribute all of processor `i`'s right values across the right group, all its left values across the left group.
        int l_send_remaining = M_l[i], r_send_remaining = M_r[i];
        for (int j = 0; j < q; j++) {
            const bool is_j_left = j < q_l; // Is processor `j` in the left group?
            const int chunk_size = is_j_left ? M_l[i] / q_l : M_r[i] / q_r; // Number of values to distribute to each group member.
            const int send_remainder = is_j_left ? M_l[i] % q_l : M_r[i] % q_r; // Number of values to distribute across group members.
            int *send_remaining = is_j_left ? &l_send_remaining : &r_send_remaining;
            all_send_counts[i][j] = std::min(*send_remaining, chunk_size + ((is_j_left ? j : j - q_l) < send_remainder ? 1 : 0));
            *send_remaining -= all_send_counts[i][j];
        }
    }

    vector<int> send_counts(q);
    for (int i = 0; i < q; i++) send_counts[i] = all_send_counts[r][i];
    vector<int> send_displs(q);
    vector<int> recv_counts(q);
    vector<int> recv_displs(q);
    for (int i = 0; i < q; i++) {
        send_displs[i] = i == 0 ? 0 : send_displs[i - 1] + send_counts[i - 1];
        recv_counts[i] = all_send_counts[i][r];
        recv_displs[i] = i == 0 ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
    }

    // Transfer the data to create two subproblems of size `m_l` and `m_r`.
    // TODO the documentation says there is no "in-place" version of "Alltoallv",
    // but I've had success just using the same buffer for both send and receive.
    // Need to try this across many test cases, but if it works, we could just resize `values_local` and use it for send/receive.
    vector<int> recv_values(recv_displs[q - 1] + recv_counts[q - 1]);
    MPI_Alltoallv(&values_local[0], &send_counts[0], &send_displs[0], MPI_INT, &recv_values[0], &recv_counts[0], &recv_displs[0], MPI_INT, comm);

    // Debug
    cout << "Processor " << r << '\n'
        << "\tpivot " << pivot << '\n'
        << "\tm_l = " << m_l << ", m_r = " << m_r << '\n'
        << "\tm_l_local = " << m_l_local << ", m_r_local = " << m_r_local << '\n'
        << "\tq_l = " << q_l << ", q_r = " << q_r << '\n'
        << "\tgroup = " << (is_left ? "left" : "right") << '\n'
        << "\tvalues_local (pre):\n\t\t" << to_string(values_local_pre) << '\n'
        << "\tvalues_local (post):\n\t\t" << to_string(values_local) << '\n'
        << "\trecv_values:\n\t\t" << to_string(recv_values) << '\n'
        << "\tsend_counts:\n\t\t" << to_string(send_counts) << '\n'
        << "\tsend_displs:\n\t\t" << to_string(send_displs) << '\n'
        << "\trecv_counts:\n\t\t" << to_string(recv_counts) << '\n'
        << "\trecv_displs:\n\t\t" << to_string(recv_displs) << '\n';

    // Copy received values into the local array.
    values_local.resize(recv_values.size());
    for (int i = 0; i < recv_values.size(); i++) values_local[i] = recv_values[i];

    // Create two new communicators corresponding to the two partitions.
    MPI_Comm new_comm;
    MPI_Comm_split(comm, is_left, r, &new_comm);

    // Recursively sort the two partitions.
    quicksort_parallel(values_local, is_left ? m_l : m_r, new_comm);

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
