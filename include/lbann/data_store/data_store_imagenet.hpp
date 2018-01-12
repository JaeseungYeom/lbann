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

#ifndef __DATA_STORE_IMAGENET_HPP__
#define __DATA_STORE_IMAGENET_HPP__

#include "lbann/data_store/generic_data_store.hpp"

namespace lbann {

/**
 * todo
 */

class data_store_imagenet : public generic_data_store {
 public:

  //! ctor
  data_store_imagenet(lbann_comm *comm, generic_data_reader *reader) :
    generic_data_store(comm, reader) {}

  //! copy ctor
  data_store_imagenet(const data_store_imagenet&) = default;

  //! operator=
  data_store_imagenet& operator=(const data_store_imagenet&) = default;

  generic_data_store * copy() const override { return new data_store_imagenet(*this); }

  //! dtor
  ~data_store_imagenet() override {}

  void setup() override;

  void get_data_buf(std::string dir, std::string filename, std::vector<unsigned char> *&buf, int tid);

 protected :

};

}  // namespace lbann

#endif  // __GENERIC_DATA_STORE_HPP__