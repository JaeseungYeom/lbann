////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
// data_reader_imagenet .hpp .cpp - data reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/utils/image.hpp"
#include "lbann/utils/file_utils.hpp"
#include <vector>

namespace lbann {

imagenet_reader::imagenet_reader(bool shuffle)
  : image_data_reader(shuffle) {
  set_defaults();
}

imagenet_reader::~imagenet_reader() {
  int num_io_threads = this->m_rng_record.size();

  std::string ofn_r = "rng-" + m_role + '-'
                    + std::to_string(this->m_comm->get_trainer_rank()) + '-'
                    + std::to_string(this->m_comm->get_rank_in_trainer()) +'-';

  std::string ofn_s = "sample-" + m_role + '-'
                    + std::to_string(this->m_comm->get_trainer_rank()) + '-'
                    + std::to_string(this->m_comm->get_rank_in_trainer()) +'-';

  for(int tid = 0; tid < num_io_threads; ++tid) {
    std::ofstream os_r(ofn_r + std::to_string(tid) + ".txt");
    const auto& record_r = this->m_rng_record[tid];
    for (const auto r: record_r) {
      os_r << std::hex << r << std::endl;
    }
    os_r.close();

    std::ofstream os_s(ofn_s + std::to_string(tid) + ".txt");
    const auto& record_s = this->m_sample_record[tid];
    for (const auto s: record_s) {
      os_s << s << std::endl;
    }
    os_s.close();
  }
}

void imagenet_reader::set_defaults() {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 1000;
}

CPUMat imagenet_reader::create_datum_view(CPUMat& X, const int mb_idx) const {
  return El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
}

bool imagenet_reader::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  El::Matrix<uint8_t> image;
  std::vector<size_t> dims;
  const std::string image_path = get_file_dir() + m_image_list[data_id].first;
  if (m_data_store != nullptr) {
    bool have_node = true;
    conduit::Node node;
    if (m_data_store->is_local_cache()) {
      if (m_data_store->has_conduit_node(data_id)) {
        const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
        node.set_external(ds_node);
      } else {
        load_conduit_node_from_file(data_id, node);
        m_data_store->set_conduit_node(data_id, node);
      }
    } else if (data_store_active()) {
      const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
      node.set_external(ds_node);
    } else if (priming_data_store()) {
      load_conduit_node_from_file(data_id, node);
      m_data_store->set_conduit_node(data_id, node);
    } else {
      if (get_role() != "test") {
        LBANN_ERROR("you shouldn't be here; please contact Dave Hysom");
      }
      if (m_issue_warning) {
        if (is_master()) {
          LBANN_WARNING("m_data_store != nullptr, but we are not retrivieving a node from the store; role: " + get_role() + "; this is probably OK for test mode, but may be an error for train or validate modes");
        }
      }
      m_issue_warning = false;
      load_image(image_path, image, dims);
      have_node = false;
    }

    if (have_node) {
      char *buf = node[LBANN_DATA_ID_STR(data_id) + "/buffer"].value();
      size_t size = node[LBANN_DATA_ID_STR(data_id) + "/buffer_size"].value();
      El::Matrix<uint8_t> encoded_image(size, 1, reinterpret_cast<uint8_t*>(buf), size);
      decode_image(encoded_image, image, dims);
    }
  } 
  
  // this block fires if not using data store
  else {
    load_image(image_path, image, dims);
  }

  auto X_v = create_datum_view(X, mb_idx);
  m_transform_pipeline.apply(image, X_v, dims, m_rng_record.at(mb_idx % m_rng_record.size()));
  if ((mb_idx % m_rng_record.size()) != (size_t) get_local_thread_idx()) {
    std::string err_msg = "thread id does not match mb_idx "
                        + std::to_string(mb_idx % m_rng_record.size())
                        + " != " + std::to_string(get_local_thread_idx());
    LBANN_ERROR(err_msg);
  }

  return true;
}

}  // namespace lbann
