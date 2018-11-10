////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _FLUX_OFFLINE_TOOL_MODE_
#include "lbann/data_readers/data_reader_flux.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/io/data_buffers/distributed_io_buffer.hpp"
#else
#include "data_reader_flux.hpp"
#endif // _FLUX_OFFLINE_TOOL_MODE_

#include "lbann/utils/file_utils.hpp" // for add_delimiter() in load()
#include <limits>     // numeric_limits
#include <algorithm>  // max_element
#include <numeric>    // accumulate
#include <functional> // multiplies
#include <type_traits>// is_same
#include <set>
#include <map>
#include <sstream>
#include <omp.h>
#include "lbann/utils/timer.hpp"
#include "lbann/utils/glob.hpp"
#include "lbann/utils/peek_map.hpp"


// This macro may be moved to a global scope
#define _THROW_LBANN_EXCEPTION_(_CLASS_NAME_,_MSG_) { \
  std::stringstream _err; \
  _err << __FILE__ << ' '  << __LINE__ << " :: " \
      << (_CLASS_NAME_) << "::" << (_MSG_); \
  throw lbann_exception(_err.str()); \
}

#define _THROW_LBANN_EXCEPTION2_(_CLASS_NAME_,_MSG1_,_MSG2_) { \
  std::stringstream _err; \
  _err << __FILE__ << ' '  << __LINE__ << " :: " \
      << (_CLASS_NAME_) << "::" << (_MSG1_) << (_MSG2_); \
  throw lbann_exception(_err.str()); \
}

// This comes after all the headers, and is only visible within the current implementation file.
// To make sure, we put '#undef _CN_' at the end of this file
#define _CN_ "data_reader_flux"

namespace lbann {

bool check_file_handle(std::ifstream* h) {
std::cerr << ((h == nullptr)? "null ptr" : (!h->is_open()? "not good": "good")) << std::endl;
  return ((h != nullptr) && h->is_open());
}

std::ifstream* open_file_handle(const std::string& n) {
  return new std::ifstream(n.c_str(), std::ifstream::in | std::ifstream::binary);
}

size_t get_file_size(std::ifstream* hnd) {
  if (!check_file_handle(hnd)) {
    return 0u;
  }

  const auto curpos = hnd->tellg();
  hnd->seekg(0, std::ios::end);
  const size_t fsize = static_cast<size_t>(hnd->tellg());
  hnd->seekg(curpos);

  return fsize;
}

size_t data_reader_flux::compute_num_samples(file_handle_t hnd) const {
  const size_t fsize = get_file_size(hnd);
  // assuming one response variable
  const unsigned num_samples = fsize / sizeof(val_t) / (m_num_independent_variables + 1u);
  if (num_samples * sizeof(val_t) * (m_num_independent_variables + 1u) != fsize) {
    std::cerr << "Corrupted file? "
              << num_samples << " * " << sizeof(val_t) << " * "
              << (m_num_independent_variables + 1u) << " != "
              << fsize << std::endl;
    return 0u;
  }
  return num_samples;
}

file_handles::~file_handles() {
  for (auto& h: m_open_input_files) {
  #if 0
    conduit::relay::io::hdf5_close_file(h.second);
  #else
    h.second->close();
    delete h.second;
    h.second = nullptr;
  #endif
  }
  m_open_input_files.clear();
}

bool file_handles::add(const std::string& fname, file_handle_t hnd) {
  auto ret1 = m_open_input_files.insert(std::pair<std::string, file_handle_t>(fname, hnd));
  auto ret2 = m_open_hdf5_handles.insert(std::pair<file_handle_t, std::string>(hnd, fname));
  return ret1.second && ret2.second;
}

file_handle_t file_handles::get(const std::string& fname) const {
  std::unordered_map<std::string, file_handle_t>::const_iterator it = m_open_input_files.find(fname);
  if (it == m_open_input_files.end()) {
    return data_input_uninitialized;
  }
  return it->second;
}

std::string file_handles::get(const file_handle_t h) const {
  return peek_map(m_open_hdf5_handles, h);
}

void data_reader_flux::renew_num_local_samples(size_t new_size) {
  m_local_samples.resize(new_size);
  m_response.resize(new_size);
}


#ifndef _FLUX_OFFLINE_TOOL_MODE_
// These methods are overriden to allow each process to load and consume a unique set of data files
bool data_reader_flux::position_valid() const {
  const bool ok = (static_cast<size_t>(m_shuffled_indices[m_current_pos]) < get_num_local_samples())
    && (m_current_pos < (int)m_shuffled_indices.size());
  if (!ok) {
    const size_t my_rank = static_cast<size_t>(m_comm->get_rank_in_model());
    std::stringstream err;
    err << "rank " << my_rank << " position invalid: m_shuffled_indices["
        << m_current_pos << "] (" << m_shuffled_indices[m_current_pos]
        << ") >= get_num_local_samples() (" << get_num_local_samples() << ")" << std::endl;
    std::cerr << err.str();
  }
  return ok;
}

void data_reader_flux::set_base_offset(const int s) {
  m_base_offset = 0;
}

void data_reader_flux::set_reset_mini_batch_index(const int s) {
  m_reset_mini_batch_index = 0;
}

int data_reader_flux::get_num_data() const {
  return m_global_num_samples_to_use;
}

void data_reader_flux::shuffle_indices() {
  // Shuffle the data
  if (m_shuffle) {
    std::shuffle(m_local_samples.begin(), m_local_samples.end(),
                 get_data_seq_generator());
  }
}

void data_reader_flux::select_subset_of_data() {

  shuffle_indices();

  const size_t count = get_absolute_sample_count();
  const double use_percent = get_use_percent();
  if (count == 0u and use_percent == 0.0) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: data_reader_flux::select_subset_of_data() get_use_percent() "
        + "and get_absolute_sample_count() are both zero; exactly one "
        + "must be zero");
  }
  if (!(count == 0u or use_percent == 0.0)) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: data_reader_flux::select_subset_of_data() get_use_percent() "
        "and get_absolute_sample_count() are both non-zero; exactly one "
        "must be zero");
  }

  if (count != 0u) {
    if(count > get_num_local_samples()) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: data_reader_flux::select_subset_of_data() - absolute_sample_count=" +
        std::to_string(count) + " is > get_num_local_samples()=" +
        std::to_string(get_num_local_samples()));
    }
    renew_num_local_samples(get_absolute_sample_count());
  }

  if (use_percent) {
    renew_num_local_samples(get_use_percent()*get_num_local_samples());
  }

  long unused = get_validation_percent()*get_num_local_samples();
  long use_me = get_num_local_samples() - unused;
  if (unused > 0) {
      m_unused_samples = sample_map_t(m_local_samples.begin() + use_me, m_local_samples.end());
      renew_num_local_samples(use_me);
  }

  if(!m_shuffle) {
    std::sort(m_local_samples.begin(), m_local_samples.end());
    std::sort(m_unused_samples.begin(), m_unused_samples.end());
  }
}

void data_reader_flux::use_unused_index_set() {
  m_local_samples.swap(m_unused_samples);
  m_unused_samples.clear();
  m_unused_samples.shrink_to_fit();
  adjust_num_samples_to_use();
}

void data_reader_flux::set_io_buffer_type(const std::string io_buffer) {
  m_io_buffer_type = io_buffer;
}

void data_reader_flux::set_open_file_handles(std::shared_ptr<file_handles>& f) {
  m_open_input_files = f;
}

std::shared_ptr<file_handles>& data_reader_flux::get_open_file_handles() {
  return m_open_input_files;
}

int data_reader_flux::compute_max_num_parallel_readers() {
  if (m_io_buffer_type == "distributed") {
    // Use a sufficiently large data set size for the time being, and
    // check if it is ok when the actual size of data is available later
    long data_set_size = 2 * get_mini_batch_size() * m_comm->get_num_models() * get_num_parallel_readers();
    set_num_parallel_readers(distributed_io_buffer::compute_max_num_parallel_readers(
                             data_set_size, get_mini_batch_size(),
                             get_num_parallel_readers(), get_comm()));
    set_sample_stride(1);
    set_iteration_stride(get_num_parallel_readers());
  } else if (m_io_buffer_type == "partitioned") {
    set_num_parallel_readers(partitioned_io_buffer::compute_max_num_parallel_readers(
                             0, get_mini_batch_size(),
                             get_num_parallel_readers(), get_comm()));
    set_sample_stride(get_num_parallel_readers());
    set_iteration_stride(1);
  } else {
    _THROW_LBANN_EXCEPTION_(get_type(), " unknown io_buffer type: " + m_io_buffer_type);
  }
  return get_num_parallel_readers();
}

bool data_reader_flux::check_num_parallel_readers(long data_set_size) {
  if (m_io_buffer_type == "distributed") {
    const bool too_many_readers = !distributed_io_buffer::check_num_parallel_readers(data_set_size, get_mini_batch_size(), get_num_parallel_readers(), m_comm);
    if (too_many_readers) {
      if(m_comm->am_world_master()) {
        std::string err =
          "The training data set size " + std::to_string(data_set_size)
          + " is too small for the number of parallel readers "
          + std::to_string(get_num_parallel_readers());
        _THROW_LBANN_EXCEPTION_(get_type(), err);
        return false;
      }
    }
  }
  return true;
}
#else // _FLUX_OFFLINE_TOOL_MODE_
void data_reader_flux::set_num_samples(size_t ns) {
  m_global_num_samples_to_use = ns;
  renew_num_local_samples(ns);
  m_num_samples = ns;
}
#endif // _FLUX_OFFLINE_TOOL_MODE_

data_reader_flux::data_reader_flux(bool shuffle)
  : generic_data_reader(shuffle) {
  set_defaults();
}

void data_reader_flux::copy_members(const data_reader_flux& rhs) {
  m_is_data_loaded = rhs.m_is_data_loaded;
  m_num_independent_variables = rhs.m_num_independent_variables;
  m_local_samples = rhs.m_local_samples;
  m_unused_samples = rhs.m_unused_samples;
  m_global_num_samples_to_use = rhs.m_global_num_samples_to_use;
  m_io_buffer_type = rhs.m_io_buffer_type;
  m_open_input_files = rhs.m_open_input_files;
  m_response = rhs.m_response;
}

data_reader_flux::data_reader_flux(const data_reader_flux& rhs)
  : generic_data_reader(rhs) {
  copy_members(rhs);
}

data_reader_flux& data_reader_flux::operator=(const data_reader_flux& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  generic_data_reader::operator=(rhs);

  copy_members(rhs);

  return (*this);
}

data_reader_flux::~data_reader_flux() {
}

void data_reader_flux::set_defaults() {
  m_is_data_loaded = false;
  m_num_independent_variables = 0u;
  m_local_samples.clear();
  m_global_num_samples_to_use = 0ul;
  m_io_buffer_type = "";
  m_open_input_files = nullptr;
  m_response.clear();
}

std::string data_reader_flux::get_description() const {
  std::string ret = std::string("data_reader_flux:\n")
    + " - num independent vars: " + std::to_string(m_num_independent_variables) + "\n"
    + " - num samples: " + std::to_string(m_global_num_samples_to_use) + "\n";
  return ret;
}

void data_reader_flux::set_num_independent_variables(size_t n) {
  m_num_independent_variables = n;
}

size_t data_reader_flux::get_num_independent_variables() const {
  return m_num_independent_variables;
}

#ifndef _FLUX_OFFLINE_TOOL_MODE_
void data_reader_flux::determine_num_samples_to_use() {
  // The meaning of m_first_n as well as absolute_sample_count is slightly
  // different in this data reader as it represents the first n local samples
  // instead of the first n global samples.
  if (m_first_n > 0) {
    const size_t num_samples = std::min(static_cast<size_t>(m_first_n), get_num_local_samples());
    renew_num_local_samples(num_samples); // this does not work with unordered_map but with vector
  }

  select_subset_of_data();
  adjust_num_samples_to_use();
}

void data_reader_flux::adjust_num_samples_to_use() {
  const size_t num_local_samples = get_num_local_samples();

  const int my_rank = m_comm->get_rank_in_model();
  const int num_readers = get_num_parallel_readers();

  // Find the minimum of the number of valid samples locally available
  unsigned long long n_loc = static_cast<unsigned long long>(num_local_samples);
  unsigned long long n_min = static_cast<unsigned long long>(num_local_samples);

  if (my_rank >= num_readers) {
    n_loc = std::numeric_limits<unsigned long long>::max();
    n_min = std::numeric_limits<unsigned long long>::max();
  }

  m_comm->model_allreduce(&n_loc, 1, &n_min, El::mpi::MIN);

  // Find the first rank that has the minimum number of valid samples
  int rank_tmp_1st = (n_loc == n_min)? my_rank : num_readers;
  int rank_min_1st;
  m_comm->model_allreduce(&rank_tmp_1st, 1, &rank_min_1st, El::mpi::MIN);

  // Determine the number of samples to use
  m_global_num_samples_to_use = static_cast<size_t>(n_min * num_readers + rank_min_1st);
  if (m_global_num_samples_to_use == static_cast<size_t>(0u)) {
    _THROW_LBANN_EXCEPTION_(get_type(), "No valid sample found.");
  }

  auto local_num_samples_to_use = (my_rank < rank_min_1st)? (n_min+1) : n_min;
  if (my_rank >= num_readers) {
    local_num_samples_to_use = 0u;
  }
  renew_num_local_samples(local_num_samples_to_use);

  // Compute data yield
  unsigned long long n_valid_local = num_local_samples;
  unsigned long long n_valid_global = 0u;
  m_comm->model_allreduce(&n_valid_local, 1, &n_valid_global, El::mpi::SUM);

  if (is_master()) {
    const double yield = static_cast<double>(m_global_num_samples_to_use)/n_valid_global;
    std::cout << "\nData yield: " << yield << std::endl;
  }

  check_num_parallel_readers(static_cast<long>(m_global_num_samples_to_use));
  populate_shuffled_indices(m_global_num_samples_to_use);
}

void data_reader_flux::populate_shuffled_indices(const size_t num_samples) {
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(num_samples);

  int s = 0;
  if (m_io_buffer_type == "partitioned") {
    const size_t s_stride = static_cast<size_t>(get_sample_stride());
    for(size_t n = 0u; n < m_shuffled_indices.size() ; n += s_stride) {
      for(size_t r = 0u; (r < s_stride) && (n+r < m_shuffled_indices.size()); ++r) {
        m_shuffled_indices[n+r] = s;
      }
      ++s;
    }
  } else if (m_io_buffer_type == "distributed") {
    const int num_readers = get_iteration_stride();
    const int mb_size = get_mini_batch_size();
    for(size_t n = 0u; n < m_shuffled_indices.size(); ) {
      for(int r = 0; r < num_readers; r++) {
        for(int m = 0, si = s; (m < mb_size) && (n < m_shuffled_indices.size()); ++m) {
          m_shuffled_indices[n++] = si++;
        }
      }
      s += mb_size;
    }
  }
}

void data_reader_flux::load() {
  const std::string data_dir = add_delimiter(get_file_dir());
  const std::string conduit_file_name = get_data_filename();
  const std::string pattern = data_dir + conduit_file_name;
  std::vector<std::string> filenames = glob(pattern);
  if (filenames.size() < 1) {
    _THROW_LBANN_EXCEPTION_(get_type(), " failed to get data filenames");
  }

  // Shuffle the file names
  if (is_shuffled()) {
    std::shuffle(filenames.begin(), filenames.end(), get_data_seq_generator());
  }

  const size_t my_rank = static_cast<size_t>(m_comm->get_rank_in_model());
  const size_t num_readers = static_cast<size_t>(compute_max_num_parallel_readers());

  // handle data partitioning among models (e.g., for LTFB)
  if (m_is_partitioned) {
    const size_t one_more = filenames.size() % m_num_partitions;
    const size_t min_num_files_per_partition = filenames.size()/static_cast<size_t>(m_num_partitions);
    if (min_num_files_per_partition == 0u) {
      _THROW_LBANN_EXCEPTION_(get_type(), "Insufficient number of files for the number of models.");
    }
    const size_t p = static_cast<size_t>(m_my_partition);
    const size_t idx_start = min_num_files_per_partition * p
                           + ((p >= one_more)? one_more : p);

    const size_t idx_end = idx_start + min_num_files_per_partition
                           + ((p < one_more)? 1u : 0u);
    std::vector<std::string> filenames_partitioned(filenames.begin()+idx_start, filenames.begin()+idx_end);
    filenames = filenames_partitioned;
  }
  const size_t num_files_to_load =
    (m_max_files_to_load > 0u)? std::min(m_max_files_to_load, filenames.size()) : filenames.size();

  filenames.resize(num_files_to_load);

  double tm1 = get_time();

  // Reserve m_local_samples
  const size_t max_num_files_to_load_per_rank = (num_files_to_load + num_readers - 1u) / num_readers;
  bool valid_samples_reserved = false;
  size_t idx = static_cast<size_t>(0ul);

  for (size_t n = my_rank; (n < num_files_to_load) && (my_rank < num_readers); n += num_readers) {
    load_a_file(filenames[n], idx);
    if (!valid_samples_reserved) {
      // reserve the sufficient capacity estimated assuming that files have the same number of samples
      m_local_samples.reserve(m_local_samples.size() * (max_num_files_to_load_per_rank + 1u));
      valid_samples_reserved = true;
    }
    if (is_master()) {
      std::cerr << "time to load: " << n + num_readers << " files: " << get_time() - tm1 << std::endl;
    }
  }
  if (is_master()) {
    std::cerr << "time to load data files: " << get_time() - tm1
              << "  number of valid local samples at the master rank: "
              << m_local_samples.size() << std::endl;
  }

  determine_num_samples_to_use();
  m_response.clear();
  m_response.resize(get_num_local_samples());

  if (is_master()) {
    std::cout << std::endl << get_description() << std::endl << std::endl;
  }
}
#endif // _FLUX_OFFLINE_TOOL_MODE_

/// populates the list of samples and returns the number of samples
size_t data_reader_flux::populate_sample_list(file_handle_t input_hnd) {
  const size_t num_new_samples = compute_num_samples(input_hnd);
  // Here the sample name is the offset in the file such that the sample can be located by it
  for (size_t s = 0; s < num_new_samples; ++s) {
    m_local_samples.push_back(sample_locator_t(s, input_hnd));
  }
  return num_new_samples;
}

void data_reader_flux::load_a_file(const std::string file_path, size_t& idx) {
  if (!check_if_file_exists(file_path)) {
    _THROW_LBANN_EXCEPTION_(get_type(), " failed to open " + file_path);
  }
#ifndef _FLUX_OFFLINE_TOOL_MODE_
  const size_t my_rank = static_cast<size_t>(m_comm->get_rank_in_model());
  std::cerr << ("rank "  + std::to_string(my_rank) + " loading: " + file_path) << std::endl;
#else
  std::cerr << "loading: " << file_path << std::endl;
#endif

  file_handle_t input_hnd = open_file_handle(file_path);

  if (!m_open_input_files) {
    m_open_input_files = std::make_shared<file_handles>();
  }
  m_open_input_files->add(file_path, input_hnd);
  if (!check_file_handle(input_hnd)) {
    _THROW_LBANN_EXCEPTION_(get_type(), std::string("cannot add invalid file handle for ") + file_path);
  }

  populate_sample_list(input_hnd);

  idx = m_local_samples.size();
  if (is_master()) {
    std::cerr << "data_reader_flux::load_a_file: num samples: "
              << m_local_samples.size() << std::endl;
  }

  m_is_data_loaded = true;
}


size_t data_reader_flux::get_num_local_samples() const {
  return m_local_samples.size();
}

const data_reader_flux::sample_map_t& data_reader_flux::get_valid_local_samples() const {
  return m_local_samples;
}

const data_reader_flux::sample_map_t& data_reader_flux::get_valid_local_samples_unused() const {
  return m_unused_samples;
}

int data_reader_flux::get_linearized_data_size() const {
  return static_cast<int>(m_num_independent_variables);
}

int data_reader_flux::get_linearized_response_size() const {
  return 1;
}

const std::vector<int> data_reader_flux::get_data_dims() const {
    return std::vector<int>(m_num_independent_variables);
}

bool data_reader_flux::check_sample_id(const size_t sample_id) const {
  return (sample_id < m_local_samples.size());
}

bool data_reader_flux::load_sample(const size_t i, std::vector<val_t>& sample) const {
  const sample_name_t& sample_offset = m_local_samples[i].first;
  file_handle_t h = m_local_samples[i].second;
  if (!check_file_handle(h)) {
    _THROW_LBANN_EXCEPTION_(get_type(), "Invalid file handle for sample " + std::to_string(i));
    return false;
  }

  const size_t sample_size = static_cast<size_t>(get_linearized_data_size() + get_linearized_response_size());
  sample.clear();
  sample.resize(sample_size);
  const size_t byte_size = sample_size*sizeof(val_t);
  h->seekg(sample_offset * byte_size);
  h->read(reinterpret_cast<char*>(sample.data()), byte_size);

  return true;
}

bool data_reader_flux::fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) {
  int num_vals = X.Height();
  if (static_cast<size_t>(data_id) >= m_local_samples.size()) {
    _THROW_LBANN_EXCEPTION_(get_type(), " invalid sample index: " \
                            + std::to_string(data_id) + " >= " \
                            + std::to_string(m_local_samples.size()));
    return false;
  }

  std::vector<val_t> sample;
  load_sample(data_id, sample);

  if (sample.size() != static_cast<size_t>(num_vals + 1)) {
    _THROW_LBANN_EXCEPTION_(get_type(), " invalid sample size: " \
                            + std::to_string(sample.size()) + " != " \
                            + std::to_string(num_vals));
  }

  for (int r = 0; r < num_vals; r++) {
    X.Set(r, mb_idx, sample.at(r));
  }
  m_response[data_id] = sample.back();

  return true;
}

bool data_reader_flux::fetch_response(CPUMat& X, int data_id, int mb_idx, int tid) {
  X.Set(0, mb_idx, m_response[data_id]);
  return true;
}

} // end of namespace lbann

#undef _CN_
