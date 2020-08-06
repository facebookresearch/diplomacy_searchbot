#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "game.h"
#include "orders_encoder.h"

namespace py = pybind11;

namespace dipcc {

// Job Types
enum ThreadPoolJobType { STEP, ENCODE };

// Used for ENCODE job type
struct EncodingArrayPointers {
  float *x_board_state;
  float *x_prev_state;
  long *x_prev_orders;
  float *x_season;
  float *x_in_adj_phase;
  float *x_build_numbers;
  int8_t *x_loc_idxs;
  int32_t *x_possible_actions;
  int32_t *x_max_seq_len;
};

// Struct for all job types
struct ThreadPoolJob {
  ThreadPoolJobType job_type;
  std::vector<Game *> games;
  std::vector<EncodingArrayPointers> encoding_array_pointers;

  ThreadPoolJob() {}
  ThreadPoolJob(ThreadPoolJobType type) : job_type(type) {}
};

class ThreadPool {
public:
  ThreadPool(size_t n_threads,
             std::unordered_map<std::string, int> order_vocabulary_to_idx,
             int max_order_cands);
  ~ThreadPool();

  const OrdersEncoder &get_orders_encoder() const { return orders_encoder_; }

  // Call game.process() on each of the games. Blocks until all process()
  // functions have exited.
  void process_multi(std::vector<Game *> &games);

  // Fill a list of pre-allocated DataFields objects with the games' input
  // encodings
  void encode_inputs_multi(
      std::vector<Game *> &games, std::vector<float *> &x_board_state,
      std::vector<float *> &x_prev_state, std::vector<long *> &x_prev_orders,
      std::vector<float *> &x_season, std::vector<float *> &x_in_adj_phase,
      std::vector<float *> &x_build_numbers, std::vector<int8_t *> &x_loc_idxs,
      std::vector<int32_t *> &x_possible_actions,
      std::vector<int32_t *> &x_max_seq_len);

private:
  /////////////
  // Methods //
  /////////////

  // Worker thread entrypoint function
  void thread_fn();

  // Top-level job handler
  void thread_fn_do_job_unsafe(ThreadPoolJob &);

  // Job handler methods
  void do_job_step(ThreadPoolJob &);
  void do_job_encode(ThreadPoolJob &);

  //////////
  // Data //
  //////////

  std::vector<ThreadPoolJob> jobs_;
  std::mutex mutex_;
  std::condition_variable cv_in_;
  std::condition_variable cv_out_;
  size_t unfinished_jobs_;
  bool time_to_die_ = false;

  std::vector<std::thread> threads_;
  const OrdersEncoder orders_encoder_;
};

} // namespace dipcc
