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
// lbann_proto.cpp - prototext application
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf_utils.hpp"
#include <cstdlib>

using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  if (master) {
    std::cout << "\n\n==============================================================\n"
              << "STARTING lbann with this command line:\n";
    for (int j=0; j<argc; j++) {
      std::cout << argv[j] << " ";
    }
    std::cout << std::endl << std::endl;
  }

#ifdef EL_USE_CUBLAS
  El::GemmUseGPU(32,32,32);
#endif

  try {
    // Initialize options db (this parses the command line)
    options *opts = options::get();
    opts->init(argc, argv);
    if (opts->has_string("h") or opts->has_string("help") or argc == 1) {
      print_help(comm);
      finalize(comm);
      return 0;
    }

    //this must be called after call to opts->init();
    if (!opts->has_bool("disable_signal_handler")) {
      std::string file_base = (opts->has_bool("stack_trace_to_file") ?
                               "stack_trace" : "");
      stack_trace::register_signal_handler(file_base);
    }

    //to activate, must specify --st_on on cmd line
    stack_profiler::get()->activate(comm->get_rank_in_world());

    std::stringstream err;

    std::vector<lbann_data::LbannPB *> pbs;
    protobuf_utils::load_prototext(master, argc, argv, pbs);

    model *model_1 = build_model_from_prototext(argc, argv, *(pbs[0]),
                                                comm, true); //D1 solver
    //hack, overide model name to make reporting easy, what can break?"
    model *model_2 = nullptr; //G1 solver
    model *model_3 = nullptr; //G2 solver

    //Support for autoencoder models
    model *ae_model = nullptr;
    model *ae_cycgan_model = nullptr; //contain layer(s) from (cyc)GAN

    if (pbs.size() > 1) {
      model_2 = build_model_from_prototext(argc, argv, *(pbs[1]),
                                           comm, false);
    }

    if (pbs.size() > 2) {
      model_3 = build_model_from_prototext(argc, argv, *(pbs[2]),
                                           comm, false);
    }

    if (pbs.size() > 3) {
      ae_model = build_model_from_prototext(argc, argv, *(pbs[3]),
                                           comm, false);
    }

    if (pbs.size() > 4) {
      ae_cycgan_model = build_model_from_prototext(argc, argv, *(pbs[4]),
                                           comm, false);
    }

    const lbann_data::Model pb_model = pbs[0]->model();
    const lbann_data::Model pb_model_2 = pbs[1]->model();
    const lbann_data::Model pb_model_3 = pbs[2]->model();

    //Optionally pretrain autoencoder
    //@todo: explore joint-train of autoencoder as alternative
    if(ae_model != nullptr) {
      if(master) std::cout << " Pre-train autoencoder " << std::endl;
      const lbann_data::Model pb_model_4 = pbs[3]->model();
      ae_model->train(pb_model_4.num_epochs());
      auto ae_weights = ae_model->get_weights();
      model_1->copy_trained_weights_from(ae_weights);
      model_2->copy_trained_weights_from(ae_weights);
      model_3->copy_trained_weights_from(ae_weights);
      ae_cycgan_model->copy_trained_weights_from(ae_weights);
    }

    //Train cycle GAN
    int super_step = 1;
    int max_super_step = pb_model.super_steps();
    while (super_step <= max_super_step) {
      if (master)  std::cerr << "\nSTARTING train - discriminator (D1 & D2) models at step " << super_step <<"\n\n";
      model_1->train( super_step*pb_model.num_epochs(),pb_model.num_batches());

      if(master) std::cout << " Copy all trained weights from discriminator to G1 and train/freeze as appropriate " << std::endl;
      auto model1_weights = model_1->get_weights();
      model_2->copy_trained_weights_from(model1_weights);
      if (master) std::cerr << "\n STARTING train - G1 solver model at step " << super_step << " \n\n";
      model_2->train( super_step*pb_model_2.num_epochs(),pb_model_2.num_batches());
      // Evaluate model on test set
      model_2->evaluate(execution_mode::testing,pb_model_2.num_batches());

      if(master) std::cout << " Copy all trained weights from discriminator to G2 and train/freeze as appropriate " << std::endl;
      model_3->copy_trained_weights_from(model1_weights);
      if (master) std::cerr << "\n STARTING train - G2 solver model at step " << super_step << " \n\n";
      model_3->train( super_step*pb_model_3.num_epochs(),pb_model_3.num_batches());
      // Evaluate model on test set
      model_3->evaluate(execution_mode::testing,pb_model_3.num_batches());

      if(master) std::cout << " Update G1 weights " << std::endl;
      auto model2_weights = model_2->get_weights();
      model_1->copy_trained_weights_from(model2_weights);
      if(master) std::cout << " Update G2 weights " << std::endl;
      auto model3_weights = model_3->get_weights();
      model_1->copy_trained_weights_from(model3_weights);

      //Optionally evaluate on pretrained autoencoder
      if(ae_model != nullptr && ae_cycgan_model != nullptr){
        //if(master) std::cout << " Copy trained weights from autoencoder to autoencoder proxy" << std::endl;
        //ae_cycgan_model->copy_trained_weights_from(ae_weights);
        if(master) std::cout << " Copy trained weights from cycle GAN" << std::endl;
        ae_cycgan_model->copy_trained_weights_from(model2_weights);
        if(master) std::cout << " Evaluate pretrained autoencoder" << std::endl;
        ae_cycgan_model->evaluate(execution_mode::testing);
      }

      super_step++;
    }

    //has no affect unless option: --st_on was given
    stack_profiler::get()->print();

    delete model_1;
    if (model_2 != nullptr) {
      delete model_2;
    }
    if (model_3 != nullptr) {
      delete model_3;
    }
    if (ae_model != nullptr) {
      delete ae_model;
    }
    if (ae_cycgan_model != nullptr) {
      delete ae_cycgan_model;
    }
    for (auto t : pbs) {
      delete t;
    }

  } catch (std::exception& e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  }

  // Clean up
  finalize(comm);
  return EXIT_SUCCESS;
}
