#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <locale>
#include "sample_list_jag.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"
#include <deque>
#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <unordered_set>
#include <memory>

namespace lbann {

inline sample_list_header::sample_list_header()
 : m_is_exclusive(false), m_sample_count(0u), m_num_files(0u), m_file_dir("") {
}

inline bool sample_list_header::is_exclusive() const {
  return m_is_exclusive;
}

inline size_t sample_list_header::get_sample_count() const {
  return m_sample_count;
}

inline size_t sample_list_header::get_num_files() const {
  return m_num_files;
}

inline const std::string& sample_list_header::get_file_dir() const {
  return m_file_dir;
}


inline void sample_list_jag::set_num_partitions(size_t n) {
  if (n == 0) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: number of partitions must be a positive number ("
                          + std::to_string(n) + ")");
  }
  clear();
  m_num_partitions = n;
  if (!m_sample_list.empty()) {
    get_sample_range_per_part();
  }
}

inline void sample_list_jag::load(const std::string& samplelist_file) {
  std::ifstream istr(samplelist_file);
  get_samples_per_file(istr);
  istr.close();
  get_sample_range_per_part();
}

/*
inline void sample_list_jag::shuffle(gen) {
// TODO: shuffling in a separate function, and need to pass random number generator gen
  if (m_shuffled_indices.size() != m_sample_list.size()) {
    m_shuffled_indices.resize(m_sample_list.size());
    std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  }
  if (m_shuffle) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(), gen);
  }
}
*/


inline void sample_list_jag::load_from_string(const std::string& samplelist) {
  std::istringstream istr(samplelist);
  get_samples_per_file(istr);
  get_sample_range_per_part();
}


inline std::string sample_list_jag::read_header_line(std::istream& istrm, const std::string& info) const {
  if (!istrm.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: unable to read the header line of sample list for " + info);
  }

  std::string line;
  std::getline(istrm, line);

  if (line.empty()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: unable to read the header line of sample list for " + info);
  }
  return line;
}


inline sample_list_header sample_list_jag::read_header(std::istream& istrm) const {
  sample_list_header hdr;

  std::string line1 = read_header_line(istrm, "the exclusiveness");
  std::stringstream header1(line1);

  std::string line2 = read_header_line(istrm, "the number of samlpes and the number of files");
  std::stringstream header2(line2);

  std::string line3 = read_header_line(istrm, "the data file directory");
  std::stringstream header3(line3);

  std::string sample_list_type;
  header1 >> sample_list_type;
  std::for_each(sample_list_type.begin(), sample_list_type.end(), [](char& c){ c = std::toupper(c); });

  const std::string type_exclusive = "EXCLUSIVE";
  size_t found = sample_list_type.find(type_exclusive);

  if (found != std::string::npos) {
    std::cout << "Exclusive (" + sample_list_type + ") sample list" << std::endl;
    hdr.m_is_exclusive = true;
  } else {
    std::cout << "Inclusive sample list" << std::endl;
    hdr.m_is_exclusive = false;
  }

  header2 >> hdr.m_sample_count;
  header2 >> hdr.m_num_files;

  header3 >> hdr.m_file_dir;

  if (hdr.get_file_dir().empty() || !check_if_dir_exists(hdr.get_file_dir())) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: data root directory '" + hdr.get_file_dir() + "' does not exist.");
  }

  return hdr;
}


inline void sample_list_jag::read_exclusive_list(std::istream& istrm) {
  const std::string whitespaces(" \t\f\v\n\r");
  size_t cnt_files = 0u;
  std::string line;

  while (std::getline(istrm, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) { // empty line
      continue;
    }
    if (cnt_files++ >= m_header.get_num_files()) {
      break;
    }

    std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;
    size_t valid_samples;
    size_t invalid_samples;
    std::unordered_set<size_t> excluded_sample_indices;

    sstr >> filename >> valid_samples >> invalid_samples;

    const std::string conduit_file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(conduit_file_path)) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                            + " :: data file '" + filename + "' does not exist.");
    }

    excluded_sample_indices.reserve(valid_samples + invalid_samples);

    while(!sstr.eof()) {
      size_t index;
      sstr >> index;
      excluded_sample_indices.insert(index);
    }

    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_file_path );

    if (hdf5_file_hnd <= static_cast<hid_t>(0)) {
      std::cout << "Opening the file didn't work" << std::endl;
      continue; // skipping the file
    }

    std::vector<std::string> sample_names;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", sample_names);

    size_t i = 0u;
    for(auto s : sample_names) {
      std::unordered_set<size_t>::const_iterator found = excluded_sample_indices.find(i++);
      if (found != excluded_sample_indices.cend()) {
        continue;
      }

      m_sample_list.emplace_back(filename, s);
    }
  }

  m_header.m_is_exclusive = false;
}


inline void sample_list_jag::read_inclusive_list(std::istream& istrm) {
  const std::string whitespaces(" \t\f\v\n\r");
  size_t cnt_files = 0u;
  std::string line;

  while (std::getline(istrm, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) { // empty line
      continue;
    }
    if (cnt_files++ >= m_header.get_num_files()) {
      break;
    }

    std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;
    size_t valid_samples;
    size_t invalid_samples;

    sstr >> filename >> valid_samples >> invalid_samples;

    const std::string conduit_file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(conduit_file_path)) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                            + " :: data file '" + filename + "' does not exist.");
    }

    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_file_path );

    if (hdf5_file_hnd <= static_cast<hid_t>(0)) {
      std::cerr << "Opening the file didn't work" << std::endl;
      continue; // skipping the file
    }

    std::vector<std::string> sample_names;

    while(!sstr.eof()) {
      std::string sample_name;;
      sstr >> sample_name;
      m_sample_list.emplace_back(filename, sample_name);
    }
  }
}


inline size_t sample_list_jag::get_samples_per_file(std::istream& istrm) {
  m_header = read_header(istrm);
  m_sample_list.reserve(m_header.get_sample_count());

  if (m_header.is_exclusive()) {
    read_exclusive_list(istrm);
  } else {
    read_inclusive_list(istrm);
  }

  return m_sample_list.size();
}


inline void sample_list_jag::get_sample_range_per_part() {
  // Populates m_sample_range_per_part, requires the total number of samples
  // and number of partitions are known.
  const size_t total = static_cast<size_t>(m_sample_list.size());
  const size_t one_more = total % m_num_partitions;
  const size_t min_per_partition = total/m_num_partitions;

  if (min_per_partition == 0u) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
          + " :: insufficient number of samples " + std::to_string(total)
          + " for each partition " + std::to_string(m_num_partitions) + " to have at least one.");
  }

  m_sample_range_per_part.clear();
  m_sample_range_per_part.resize(m_num_partitions+1u);

  #pragma omp parallel for
  for (size_t p = 0u; p < m_num_partitions; ++p) {
    const size_t r_start = min_per_partition * p + ((p >= one_more)? one_more : p);
    const size_t r_end = r_start + min_per_partition + ((p < one_more)? 1u : 0u);
    m_sample_range_per_part[p+1] = r_end;
  }
}


inline void sample_list_jag::get_sample_range_per_part(const size_t p, size_t& sid_start, size_t& sid_end) const{
  // Populates m_sample_range_per_part, requires the total number of samples
  // and number of partitions are known.
  const size_t total = static_cast<size_t>(m_sample_list.size());
  const size_t one_more = total % m_num_partitions;
  const size_t min_per_partition = total/m_num_partitions;

  if (min_per_partition == 0u) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
          + " :: insufficient number of samples for each partition to have at least one.");
  }

  sid_start = min_per_partition * p + ((p >= one_more)? one_more : p);
  sid_end = sid_start + min_per_partition + ((p < one_more)? 1u : 0u);
}


inline void sample_list_jag::clear() {
  m_num_partitions = 1u;
  m_sample_list.clear();
  m_sample_range_per_part.clear();
}


inline void sample_list_jag::write_header(std::string& sstr) const {
  // The first line indicate if the list is exclusive or inclusive
  // The next line contains the number of samples and the number of files, which are the same in this caes
  // The next line contains the root data file directory

  sstr += (m_header.is_exclusive()? "EXCLUSIVE\n" : "INCLUSIVE\n");
  sstr += std::to_string(m_sample_list.size()) + ' ' + std::to_string(m_sample_list.size()) + '\n';
  sstr += m_header.get_file_dir() + '\n';
}


inline bool sample_list_jag::to_string(size_t p, std::string& sstr) const {
  if (p >= m_num_partitions) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
          + " :: partition id is out of range.");
    return false;
  }

  size_t i_begin, i_end;
  get_sample_range_per_part(p, i_begin, i_end);

  if (i_begin > i_end) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
          + " :: incorrect partition range.");
    return false;
  }

  samples_t::const_iterator it_begin = m_sample_list.cbegin();
  samples_t::const_iterator it_end = m_sample_list.cbegin();
  std::advance(it_begin, i_begin);
  std::advance(it_end, i_end);

  sstr.clear();

  // reserve the string to hold the entire sample lit
  size_t estimated_len = 30 + 42 + m_header.get_file_dir().size() + 1;
  if (i_begin < i_end) {
    estimated_len += static_cast<size_t>(1.5 * (i_end - i_begin) * (it_begin->first.size() + it_begin->second.size() + 6));
    sstr.reserve(estimated_len);
  }

  // write the list header
  write_header(sstr);

  // write the list body
  for (samples_t::const_iterator it = it_begin; it != it_end; ++it) {
    sstr += it->first + " 1 0 " + it->second + '\n';
  }

  std::cerr << "estimated size vs actual size: " << estimated_len << ' ' << sstr.size() << std::endl;
  std::cerr << "num samples: " << i_end - i_begin << " samples of rank " << p << std::endl;
  return true;
}


inline bool sample_list_jag::to_string(std::string& sstr) const {
  size_t total_len = 0u;
  std::vector<std::string> strvec(m_num_partitions);
  bool ok = true;

  for(size_t p=0u; (p < m_num_partitions) && ok; ++p) {
    ok = to_string(p, strvec[p]);
    total_len += strvec[p].size();
  }

  if (!ok) {
    return false;
  }

  sstr.clear();
  sstr.reserve(total_len);

  for(size_t p=0u; p < m_num_partitions; ++p) {
    sstr += strvec[p];
  }

  return true;
}


inline void sample_list_jag::write(size_t p, const std::string filename) const {
  std::string filename_p = modify_file_name(filename, std::to_string(p));

  std::string dir, basename;
  parse_path(filename_p, dir, basename);
  if (!dir.empty() && !check_if_dir_exists(dir)) {
    // The creation of a shared directory must be done once in a coordinated fashion
    // among the entities that have access to it. Thus, it must be done in advance
    std::cerr << "The sample list output directory (" + dir + ") does not exist" << std::endl;
    return;
  }

  std::fstream ofs(filename_p, std::fstream::out | std::fstream::binary);

  if (!ofs.good()) {
    return;
  }

  std::string buf;
  to_string(p, buf);

  ofs.write(buf.data(), buf.size()*sizeof(std::string::value_type));
  ofs.close();
}


inline void sample_list_jag::write(const std::string filename) const {
  for (size_t p = 0u; p < m_num_partitions; ++p) {
    write(p, filename);
  }
}


inline const sample_list_jag::samples_t& sample_list_jag::get_list() const {
  return m_sample_list;
}


inline const sample_list_header& sample_list_jag::get_header() const {
  return m_header;
}


struct send_request {
  int m_receiver;
  MPI_Request m_mpi_request;
  std::shared_ptr<std::string> m_data;
  unsigned long m_buf_size;

  send_request() {
    m_data = std::make_shared<std::string>();
  }

  void set_receiver(int recv) {
    m_receiver = recv;
  }

  int get_receiver() const {
    return m_receiver;
  }

  MPI_Request& request() {
    return m_mpi_request;
  }

  std::string* data() const {
    return m_data.get();
  }

  unsigned long& size() {
    m_buf_size = static_cast<unsigned long>(m_data->size());
    return m_buf_size;
  }
};


inline void handle_mpi_error(int ierr) {
  int errclass, resultlen;;
  char err_buffer[MPI_MAX_ERROR_STRING];

  if (ierr != MPI_SUCCESS) {
    MPI_Error_class(ierr, &errclass);
    if (errclass == MPI_ERR_RANK) {
      fprintf(stderr, "Invalid rank used in MPI send call\n");
      MPI_Error_string(ierr, err_buffer, &resultlen);
      fprintf(stderr,err_buffer);
      MPI_Finalize();             /* abort*/
    }
  }
}


#ifndef _JAG_OFFLINE_TOOL_MODE_
inline void distribute_sample_list(const sample_list_jag& sn,
                            std::string& my_samples,
                            lbann_comm& _comm) {
  MPI_Comm comm = _comm.get_model_comm().comm;
#else
inline void distribute_sample_list(const sample_list_jag& sn,
                            std::string& my_samples,
                            MPI_Comm& comm) {
#endif

  int num_ranks = 1;
  int my_rank = 0;
  int root_rank = 0;
  int size_tag = 0;
  int data_tag = 1;

  // TODO: replace bare MPI calls with corresponding lbann_comm wrappers
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &num_ranks);
  MPI_Errhandler_set(comm, MPI_ERRORS_RETURN);

  if (my_rank == root_rank) {

    std::deque< send_request > send_requests;

    // Start of serialization and transmission
    MPI_Barrier(comm);

    for(int i = 0; i < num_ranks; ++i) {
      if (i == root_rank) {
        sn.to_string(static_cast<size_t>(root_rank), my_samples);
        continue;
      }

      send_requests.emplace_back();
      auto& req0 = send_requests.back();
      send_requests.emplace_back();
      auto& req1 = send_requests.back();
      req0.set_receiver(i);
      req1.set_receiver(i);
      std::string& sstr = *(req1.data());

      sn.to_string(static_cast<size_t>(i), sstr);
      unsigned long& bufsize = req1.size();

      int ierr;
      ierr = MPI_Isend(reinterpret_cast<void*>(&bufsize), 1,
                       MPI_UNSIGNED_LONG, i, size_tag, comm, &(req0.request()));
      handle_mpi_error(ierr);

      ierr = MPI_Isend(reinterpret_cast<void*>(const_cast<char*>(sstr.data())), static_cast<int>(sstr.size()),
                       MPI_BYTE, i, data_tag, comm, &(req1.request()));
      handle_mpi_error(ierr);

      const int n_prev_reqs = static_cast<int>(send_requests.size() - 2);

      for (int j = 0; j < n_prev_reqs; ++j) {
        MPI_Status status;
        int flag;
        auto& req = send_requests.front();

        MPI_Test(&(req.request()), &flag, &status);

        if (!flag) {
          break;
        }
        send_requests.pop_front();
      }
    }

    for (auto& req: send_requests) {
      MPI_Status status;
      MPI_Wait(&(req.request()), &status);
    }

    send_requests.clear();
  } else {
    // Start of serialization and transmission
    MPI_Barrier(comm);

    MPI_Status status;
    int ierr;
    unsigned long bufsize = 0u;
    ierr = MPI_Recv(reinterpret_cast<void*>(&bufsize), 1,
                    MPI_UNSIGNED_LONG, root_rank, size_tag, comm, &status);
    handle_mpi_error(ierr);

    my_samples.resize(bufsize);

    ierr = MPI_Recv(reinterpret_cast<void*>(&my_samples[0]), static_cast<int>(bufsize),
                    MPI_BYTE, root_rank, data_tag, comm, &status);
    handle_mpi_error(ierr);
  }

  // End of serialization and transmission
  MPI_Barrier(MPI_COMM_WORLD);
}


} // end of namespace lbann