#include <algorithm>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <iomanip>

using std::string;
using std::to_string;
using std::vector;

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

// Fill the provided displacements array based on the counts array, both of size `n`,
// for use with MPI communication primitives.
void fill_displacements(int displs[], int counts[], int n) {
    for (int i = 0; i < n; i++) {
        displs[i] = i == 0 ? 0 : displs[i - 1] + counts[i - 1];
    }
}
string to_string(int values[], int size) {
    string s = "[";
    for (int i = 0; i < size; i++) {
        s += to_string(values[i]);
        if (i < size - 1) s += ", ";
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
  * Partition the `q` processors into two subgroups of size `q_l` and `q_r` (with at least one processor each),
    by allocating processors proportionally.
  * Assign `q_l` and `q_r` the tasks of sorting `m_l` and `m_r` values, respectively.
  * On each processor, use `M_l` and `M_r` to compute where to send its data and from where to receive its data.
  * Use an `Alltoallv` to perform the data transfer.
  * Create two new communicators corresponding to the two partitions.
  * Recursively call `quicksort_parallel` within each partition.
*/
void quicksort_parallel(vector<int> &values_local, size_t m, MPI_Comm comm) {
    int q, r; // Communicator size and rank.
    MPI_Comm_size(comm, &q);
    MPI_Comm_rank(comm, &r);

    const int m_local = values_local.size();
    if (m_local == 0) return; // Nothing to do.
    if (q == 0) throw std::runtime_error("quicksort_parallel: Empty communicator assigned " + to_string(m_local) + " values to sort.");
    // Using a serial quicksort for better benchmarking, since it's most similar to the parallel version.
    if (q == 1) return quicksort_serial(values_local, 0, m_local - 1);

    /**
      Generate a pivot and broadcast the pivot value to all processors in `comm`:
      * Generate a random number `k` between 0 and m − 1.
      * Gather the lengths each processor's local array (which could have different sizes due to partitioning) onto every processor in `comm`.
      * Compute the rank of the processor that has the pivot.
      * Broadcast the pivot value from the processor that has it.
    */

    // Use the same random seed on all processors within `comm`, so that each processor generates the same random number.
    static const int seed = 123;
    static std::mt19937 rand_gen(seed);
    static std::uniform_int_distribution<> rand_dist(0, m - 1);
    const int pivot_i = rand_dist(rand_gen);

    int M_local[q]; // Holds the lengths of each processor's local array.
    MPI_Allgather(&m_local, 1, MPI_INT, &M_local[0], 1, MPI_INT, comm);

    int pivot_r = 0; // Rank of the processor that has the pivot. All processors need to find this rank to participate in the broadcast of the pivot value.
    int pivot; // Only gets populated on the processor that has the pivot.
    int global_i = 0; // Running sum of the lengths of all processors' local arrays.
    for (int i = 0; i < q; i++) {
        if (pivot_i < global_i + M_local[i]) {
            pivot_r = i;
            if (i == r) pivot = values_local[pivot_i - global_i];
            break;
        }
        global_i += M_local[i];
    }

    MPI_Bcast(&pivot, 1, MPI_INT, pivot_r, comm);

    // Arrange local values so the left `m_l_local` values <= pivot and right `m_r_local` values > pivot.
    int i = 0, j = m_local - 1;
    while (i <= j) {
        while (i <= j && values_local[i] <= pivot) i++;
        while (i <= j && values_local[j] > pivot) j--;
        if (i <= j) std::swap(values_local[i++], values_local[j--]);
    }
    const int m_l_local = i, m_r_local = m_local - i;

    // Gather `m_l_local` and `m_r_local` from all processors into `M_l` and `M_r`,
    // and compute the total number of integers <= pivot (`m_l`) and > pivot (`m_r`).
    vector<int> M_l(q), M_r(q);
    MPI_Allgather(&m_l_local, 1, MPI_INT, &M_l[0], 1, MPI_INT, comm);
    MPI_Allgather(&m_r_local, 1, MPI_INT, &M_r[0], 1, MPI_INT, comm);
    const int m_l = std::accumulate(M_l.begin(), M_l.end(), 0);
    const int m_r = std::accumulate(M_r.begin(), M_r.end(), 0);

    /**
      Partition the `q` processors into two subproblems of sorting `m_l` and `m_r` values, allocating processors proportionally.
      Note: If `m_l` or `m_r` is 0, we still partition the processors into two groups with at least one processor in each group,
        but the single processor in the empty group will receive no elements.
      Compute the number of processors in the left/right groups by satisfying the following constraints:
        * `q_l + q_r = q` (use all processors)
        * `q_l/q_r = m_l/m_r` (assign processors proportionally to problem sizes)
        * `q_l > 0, q_r > 0` (assign at least one processor to each group)
    */
    int q_l = round(float(q * m_l) / float(m_l + m_r)); // Round to nearest integer instead of truncating.
    int q_r = q - q_l;
    if (q_l == 0) q_l = 1, q_r = q - 1;
    else if (q_r == 0) q_l = q - 1, q_r = 1;
    const bool is_left = r < q_l;

    // Find all `Alltoallv` values for this processor.
    // Share destination iterators across all source processors to round-robin sends across the left & right groups evenly.
    int send_counts[q], recv_counts[q];
    for (int i = 0; i < q; i++) send_counts[i] = recv_counts[i] = 0;

    int i_l = 0, i_r = 0;
    for (int source_q = 0; source_q < q; source_q++) {
        // Distribute all of `source_q`'s left values across the left group.
        for (int _ = 0; _ < M_l[source_q]; _++) {
            const int dest_q = i_l++ % q_l;
            if (source_q == r) send_counts[dest_q]++;
            if (dest_q == r) recv_counts[source_q]++;
        }
        // Distribute all of `source_q`'s right values across the right group.
        for (int _ = 0; _ < M_r[source_q]; _++) {
            const int dest_q = q_l + (i_r++ % q_r);
            if (source_q == r) send_counts[dest_q]++;
            if (dest_q == r) recv_counts[source_q]++;
        }
    }

    int send_displs[q], recv_displs[q];
    fill_displacements(send_displs, send_counts, q);
    fill_displacements(recv_displs, recv_counts, q);

    // Transfer the data so that first `q_l` processors have values <= pivot, and the rest have values > pivot.
    const int n_recv = recv_displs[q - 1] + recv_counts[q - 1]; // Total number of values to receive on this processor.
    int recv_values[n_recv];
    MPI_Alltoallv(&values_local[0], &send_counts[0], &send_displs[0], MPI_INT, &recv_values[0], &recv_counts[0], &recv_displs[0], MPI_INT, comm);

    // Copy received values into the local array.
    values_local.resize(n_recv);
    for (int i = 0; i < n_recv; i++) values_local[i] = recv_values[i];

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

    int p, r;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    const bool is_root = r == ROOT;

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

    // Block-distribute the array to all processors.
    vector<int> n_local_all, n_local_displs;
    if (is_root) {
        n_local_all.resize(p);
        n_local_displs.resize(p);
        for (int i = 0; i < p; i++) {
            n_local_all[i] = n / p + (i < n % p ? 1 : 0);
        }
        fill_displacements(&n_local_displs[0], &n_local_all[0], p);
    }
    vector<int> values_local(n / p + (r < n % p ? 1 : 0));
    MPI_Scatterv(&values_global[0], &n_local_all[0], &n_local_displs[0], MPI_INT, &values_local[0], values_local.size(), MPI_INT, ROOT, MPI_COMM_WORLD);

    // Run parallel quicksort. Note that this may change the size of `values_local`.
    const double start_time_s = MPI_Wtime();
    quicksort_parallel(values_local, n, MPI_COMM_WORLD);
    const double end_time_s = MPI_Wtime();

    // Gather the lengths and displacements of each processor's sorted array (which could have changed sizes due to partitioning).
    const int n_sorted = values_local.size();
    vector<int> n_sorted_all, n_sorted_displs;
    if (is_root) {
        n_sorted_all.resize(p);
        n_sorted_displs.resize(p);
    }
    MPI_Gather(&n_sorted, 1, MPI_INT, &n_sorted_all[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Gather the values to root, overwriting the `values_global` vector.
    if (is_root) fill_displacements(&n_sorted_displs[0], &n_sorted_all[0], p);
    MPI_Gatherv(&values_local[0], values_local.size(), MPI_INT,  &values_global[0], &n_sorted_all[0], &n_sorted_displs[0], MPI_INT, ROOT, MPI_COMM_WORLD);

    if (is_root) {
        const string output_file_path = argv[2];
        std::ofstream output_file_stream(output_file_path);
        if (!output_file_stream.is_open()) throw std::invalid_argument("Failed to open output file with path: " + output_file_path);

        for (int i = 0; i < n; i++) {
            output_file_stream << values_global[i] << (i == n - 1 ? "" : " ");
        }
        output_file_stream << '\n' << std::fixed << std::setprecision(6) << (end_time_s - start_time_s) * 1000.0 << '\n';
        output_file_stream.close();
    }
    MPI_Finalize();

    return 0;
}
