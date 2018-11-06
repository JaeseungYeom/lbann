#include <iostream>
#include <cstdio>
#include <string>
#include <set>
#include <vector>
#include <map>
#include "data_reader_flux.hpp"
#include "lbann/utils/glob.hpp"
#include <chrono>
#include <cstdio>
#include <algorithm>

inline double get_time() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(
           steady_clock::now().time_since_epoch()).count();
}

void show_help(int argc, char** argv);

void show_details(const lbann::data_reader_flux& dreader);

void fetch_iteration(const bool wrt_response,
                     const size_t mb_size,
                     lbann::data_reader_flux& dreader);


int main(int argc, char** argv)
{

  if (argc != 5) {
    show_help(argc, argv);
    return 0;
  }

  const std::string file_path_pattern = std::string(argv[1]);
  std::vector<std::string> filenames = lbann::glob(file_path_pattern);
  if (filenames.size() < 1u) {
    std::cerr << "Failed to get data filenames from: " + file_path_pattern << std::endl;
    return 0;
  }

  size_t num_ivars = static_cast<size_t>(atoi(argv[2]));
  bool measure_time = (atoi(argv[3]) > 0);
  bool wrt_response = (atoi(argv[4]) > 0);

  std::cout << "measure_time : " << measure_time << std::endl;
  std::cout << "wrt_response : " << wrt_response << std::endl;

  using namespace lbann;
  using namespace std;

  data_reader_flux dreader;

  dreader.set_num_independent_variables(num_ivars);

  double t_load =  get_time();
  for (const auto& data_file: filenames) {
    std::cout << "loading " << data_file << std::endl;
    size_t num_valid_samples = 0ul;

    dreader.load_conduit(data_file, num_valid_samples);
  }
  std::cout << "time to load consuit file: " << get_time() - t_load << " (sec)" << std::endl;

  show_details(dreader);

  const size_t n = dreader.get_num_valid_local_samples();
  // prepare the fake base data reader class (generic_data_reader) for mini batch data accesses
  const size_t mb_size = 64u;
  dreader.set_num_samples(n);
  dreader.set_mini_batch_size(mb_size);
  dreader.init();

  if (measure_time) {
    double t =  get_time();

    fetch_iteration(wrt_response, mb_size, dreader);

    std::cout << "time to read all the samples: " << get_time() - t << " (sec)" << std::endl;
  } else {

    fetch_iteration(wrt_response, mb_size, dreader);
  }

  return 0;
}



void show_help(int argc, char** argv) {
  std::cout << "Uasge: > " << argv[0] << " data_file num_independent_vars measure_time wrt_response" << std::endl;
  std::cout << "         - num_independent_vars: number of independent variables per sample." << std::endl;
  std::cout << "         - measure_time (1|0): whether to measure the time to read all the samples." << std::endl;
  std::cout << "         - wrt_response (1|0): whether to print out response variables." << std::endl;
}


void show_details(const lbann::data_reader_flux& dreader) {
  std::cout << "- number_of_samples: " << dreader.get_num_valid_local_samples() << std::endl;
  std::cout << "- linearized data size: " << dreader.get_linearized_data_size() << std::endl;
  std::cout << "- linearized response size: " << dreader.get_linearized_response_size() << std::endl;
}


void fetch_iteration(const bool wrt_response,
                     const size_t mb_size,
                     lbann::data_reader_flux& dreader) {

  const size_t n = dreader.get_num_valid_local_samples();
  const size_t nd = dreader.get_linearized_data_size();
  const size_t nr = dreader.get_linearized_response_size();
  const size_t n_full = (n / mb_size) * mb_size; // total number of samples in full mini batches
  const size_t n_rem  = n - n_full; // number of samples in the last partial mini batch (0 if the last one is full)

  CPUMat X;
  CPUMat Y;

  X.Resize(nd, mb_size);
  Y.Resize(nr, mb_size);

  if (wrt_response) {
    for (size_t s = 0u; s < n_full; s += mb_size) {
      std::cout << "samples [" << s << ' ' << s+mb_size << ")" << std::endl;
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);

      for (size_t c = 0; c < mb_size; ++c) {
        for(size_t r = 0; r < nr ; ++r) {
          printf("\t%e", Y(r, c));
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;

      dreader.update();
    }
    if (n_rem > 0u) {
      X.Resize(X.Height(), n_rem);
      Y.Resize(Y.Height(), n_rem);
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);

      for (size_t c = 0; c < n_rem; ++c) {
        for(size_t r = 0; r < nr ; ++r) {
          printf("\t%e", Y(r, c));
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;

      dreader.update();
    }
  } else {
    for (size_t s = 0u; s < n_full; s += mb_size) {
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);
      dreader.update();
    }
    if (n_rem > 0u) {
      X.Resize(X.Height(), n_rem);
      Y.Resize(Y.Height(), n_rem);
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);
      dreader.update();
    }
  }
}
