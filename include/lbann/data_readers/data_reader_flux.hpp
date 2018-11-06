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
////////////////////////////////////////////////////////////////////////////////

#ifndef _DATA_READER_FLUX_HPP_
#define _DATA_READER_FLUX_HPP_

#include "data_reader.hpp"
#include <string>
#include <set>
#include <unordered_map>
#include <map>
#include <memory>
#include <fstream>
#include <iostream>

namespace lbann {

using input_handle_t = std::ifstream*;
#if 0
constexpr auto data_input_uninitialized = static_cast<input_handle_t>(-1);
#else
constexpr auto data_input_uninitialized = static_cast<input_handle_t>(nullptr);
#endif

template<typename T>
bool check_input_handle(T) {
  return false;
}

template<typename T>
input_handle_t open_input_handle(const std::string& n) {
  return data_input_uninitialized;
}


/**
 * Store the handles of open hdf5 files, and close files at the end of the
 * life time of this container object.
 */
class input_handles {
 protected:
  std::unordered_map<std::string, input_handle_t> m_open_input_files;
  std::map<input_handle_t, std::string> m_open_hdf5_handles;

 public:
  ~input_handles();
  /// Add a handle that corresponds to the filename fname
  bool add(const std::string& fname, input_handle_t hnd);
  /**
   *  Returns the handle that corresponds to the given file name.
   *  Reuturns a negative value if not found.
   */
  input_handle_t get(const std::string& fname) const;

  std::string get(const input_handle_t h) const;

  /// Returns the read-only access to the internal data
  const std::unordered_map<std::string, input_handle_t>& get() const { return m_open_input_files; }
};


/**
 * Loads JAG simulation parameters and results from hdf5 files using conduit interfaces
 */
class data_reader_flux : public generic_data_reader {
 public:

  using val_t = double;
  using sample_name_t = size_t;
  using sample_locator_t = std::pair<sample_name_t, input_handle_t>;
  using sample_map_t = std::vector<sample_locator_t>; ///< valid sample map type

  data_reader_flux(bool shuffle = true);
  data_reader_flux(const data_reader_flux&);
  data_reader_flux& operator=(const data_reader_flux&);
  ~data_reader_flux() override;
  data_reader_flux* copy() const override { return new data_reader_flux(*this); }

  std::string get_type() const override {
    return "data_reader_flux";
  }

  void set_num_independent_variables(size_t n);
  size_t get_num_independent_variables() const;

  size_t populate_sample_list(input_handle_t input_hnd);

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// Load data and do data reader's chores.
  void load() override;
  /// True if the data reader's current position is valid.
  bool position_valid() const override;
  /// Return the base offset.
  void set_base_offset(const int s) override;
  /// Set the starting mini-batch index for the epoch
  void set_reset_mini_batch_index(const int s) override;
  /// Get the number of samples in this dataset.
  int get_num_data() const override;
  /// Select the appropriate subset of data based on settings.
  void select_subset_of_data() override;
  /// Replace the sample indices with the unused sample indices.
  void use_unused_index_set() override;
  /// Set the type of io_buffer that will rely on this reader
  void set_io_buffer_type(const std::string io_buffer);

  /// Set the set of open hdf5 data files
  void set_open_file_handles(std::shared_ptr<input_handles>& f);
  /// Get the set of open hdf5 data files
  std::shared_ptr<input_handles>& get_open_file_handles();
#else
  /// Load a data file
  void load_conduit(const std::string conduit_file_path, size_t& idx);
  /** Manually set m_global_num_samples_to_use and m_local_num_samples_to_use
   *  to avoid calling determine_num_samples_to_use();
   */
  void set_num_samples(size_t ns);
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Return the number of valid samples locally available
  size_t get_num_valid_local_samples() const;
  /// Allow read-only access to m_valid_samples member data
  const sample_map_t& get_valid_local_samples() const;
  /// Allow read-only access to m_unused_samples member data
  const sample_map_t& get_valid_local_samples_unused() const;

  /// Return the total linearized size of data
  int get_linearized_data_size() const override;
  /// Return the total linearized size of response
  int get_linearized_response_size() const override;
  const std::vector<int> get_data_dims() const override;

  /// Show the description
  std::string get_description() const;

 protected:
  virtual void set_defaults();
  virtual void copy_members(const data_reader_flux& rhs);

  bool fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx, int tid) override;

  size_t compute_num_samples(input_handle_t hnd) const;

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// Shuffle sample indices
  void shuffle_indices() override;
  /**
   * Compute the number of parallel readers based on the type of io_buffer,
   * the mini batch size, the requested number of parallel readers.
   * This is done before populating the sample indices.
   */
  int compute_max_num_parallel_readers();
  /**
   * Check if there are sufficient number of samples for the given number of
   * data readers with distributed io buffer, based on the number of samples,
   * the number of models and the mini batch size.
   */
  bool check_num_parallel_readers(long data_set_size);
  /// Determine the number of samples to use
  void determine_num_samples_to_use();
  /**
   * Approximate even distribution of samples by using as much samples
   * as commonly available to every data reader instead of using
   * all the available samples.
   */
  void adjust_num_samples_to_use();
  /**
   * populate the m_shuffled_indices such that each data reader can
   * access local data using local indices.
   */
  void populate_shuffled_indices(const size_t num_samples);
  /// Load a data file
  void load_conduit(const std::string conduit_file_path, size_t& idx);
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Check if the given sample id is valid
  bool check_sample_id(const size_t i) const;

 protected:
  /// Whether data have been loaded
  bool m_is_data_loaded;
  size_t m_num_independent_variables;

  /**
   * maps integers to sample IDs and the handle of the file that contains it.
   * In the future the sample IDs may not be integers; also, this map only
   * includes sample IDs that have <sample_id>/performance/success = 1
   */
  sample_map_t m_valid_samples;
  /// To support validation_percent
  sample_map_t m_unused_samples;

  /**
   * The number of local samples that are selected to use.
   * This is less than or equal to the number of valid samples locally available.
   */
  size_t m_local_num_samples_to_use;
  /**
   * The total number of samples to use.
   * This is the sum of m_local_num_samples_to_use.
   */
  size_t m_global_num_samples_to_use;

  /**
   * io_buffer type that will rely on this reader.
   * e.g. distributed_io_buffer, partitioned_io_buffer
   */
  std::string m_io_buffer_type;

  /// Shared set of the handles of open files/streams
  std::shared_ptr<input_handles> m_open_input_files;
};

} // end of namespace lbann
#endif // _DATA_READER_FLUX_HPP_
