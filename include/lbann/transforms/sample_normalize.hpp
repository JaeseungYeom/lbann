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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_TRANSFORMS_SAMPLE_NORMALIZE_HPP_INCLUDED
#define LBANN_TRANSFORMS_SAMPLE_NORMALIZE_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/**
 * Normalize to have mean 0, standard deviation 1.
 * This only works after conversion to an LBANN CPUMat.
 */
class sample_normalize : public transform {
public:
  transform* copy() const override { return new sample_normalize(*this); }

  std::string get_type() const override { return "sample_normalize"; }

  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) override;
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_SAMPLE_NORMALIZE_HPP_INCLUDED
