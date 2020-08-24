#include "thread_pool.h"
#include "checks.h"
#include "encoding.h"

using namespace std;

namespace dipcc {

ThreadPool::ThreadPool(
    size_t n_threads,
    std::unordered_map<std::string, int> order_vocabulary_to_idx,
    int max_order_cands)
    : orders_encoder_(order_vocabulary_to_idx, max_order_cands) {

  jobs_.reserve(n_threads);
  threads_.reserve(n_threads);
  for (int i = 0; i < n_threads; ++i) {
    threads_.push_back(thread(&ThreadPool::thread_fn, this));
  }
}

ThreadPool::~ThreadPool() {
  { // Locked critical section
    unique_lock<mutex> my_lock(mutex_);
    time_to_die_ = true;
  }
  cv_in_.notify_all();
  for (auto &th : threads_) {
    th.join();
  }
}

void ThreadPool::process_multi(vector<Game *> &games) {
  unique_lock<mutex> my_lock(mutex_);

  JCHECK(jobs_.size() == 0,
         "ThreadPool::process_multi called with non-empty jobs_");

  // Pack games into n_threads jobs
  int n_threads = threads_.size() > 0 ? threads_.size() : 1;
  for (int i = 0; i < n_threads; ++i) {
    jobs_.push_back(ThreadPoolJob(ThreadPoolJobType::STEP));
  }
  for (int i = 0; i < games.size(); ++i) {
    jobs_[i % n_threads].games.push_back(games[i]);
  }
  for (int i = 0; i < games.size(); ++i) {
    // Poor man's race condition elimination. Should not take so much time as
    // stepping job calls get_all_possible_orders on all produced states.
    games[i]->get_all_possible_orders();
  }

  // maybe handle in-thread
  if (threads_.size() == 0) {
    thread_fn_do_job_unsafe(jobs_[0]);
    jobs_.clear();
    return;
  }

  // Notify and wait for worker threads
  unfinished_jobs_ = jobs_.size();
  cv_in_.notify_all();
  while (unfinished_jobs_ != 0) {
    cv_out_.wait(my_lock);
  }
}

void ThreadPool::encode_inputs_multi(
    vector<Game *> &games, vector<float *> &x_board_state,
    vector<float *> &x_prev_state, vector<long *> &x_prev_orders,
    vector<float *> &x_season, vector<float *> &x_in_adj_phase,
    vector<float *> &x_build_numbers, vector<int8_t *> &x_loc_idxs,
    vector<int32_t *> &x_possible_actions, vector<int32_t *> &x_max_seq_len) {
  unique_lock<mutex> my_lock(mutex_);

  JCHECK(jobs_.size() == 0,
         "ThreadPool::encode_inputs_multi called with non-empty jobs_");
  JCHECK(games.size() == x_board_state.size(),
         "ThreadPool::encode_inputs_multi array sizes not equal");

  // Pack games into n_threads jobs
  int n_threads = threads_.size() > 0 ? threads_.size() : 1;
  for (int i = 0; i < n_threads; ++i) {
    jobs_.push_back(ThreadPoolJob(ThreadPoolJobType::ENCODE));
  }
  for (int i = 0; i < games.size(); ++i) {
    jobs_[i % n_threads].games.push_back(games[i]);
    jobs_[i % n_threads].encoding_array_pointers.push_back(
        EncodingArrayPointers{x_board_state[i], x_prev_state[i],
                              x_prev_orders[i], x_season[i], x_in_adj_phase[i],
                              x_build_numbers[i], x_loc_idxs[i],
                              x_possible_actions[i], x_max_seq_len[i]});
  }

  // maybe handle in-thread
  if (threads_.size() == 0) {
    thread_fn_do_job_unsafe(jobs_[0]);
    jobs_.clear();
    return;
  }

  // Notify and wait for worker threads
  unfinished_jobs_ = jobs_.size();
  cv_in_.notify_all();
  while (unfinished_jobs_ != 0) {
    cv_out_.wait(my_lock);
  }
}

void ThreadPool::thread_fn() {
  while (true) {
    ThreadPoolJob job;
    { // Locked critical section
      unique_lock<mutex> my_lock(mutex_);
      while (!time_to_die_ && jobs_.size() == 0) {
        cv_in_.wait(my_lock);
      }
      if (time_to_die_) {
        return;
      }
      job = jobs_.back();
      jobs_.pop_back();
    }

    // Do the job
    thread_fn_do_job_unsafe(job);

    // Notify done (locked critical section)
    {
      unique_lock<mutex> my_lock(mutex_);
      unfinished_jobs_--;
      if (unfinished_jobs_ == 0) {
        cv_out_.notify_all();
      }
    }
  }
}

void ThreadPool::thread_fn_do_job_unsafe(ThreadPoolJob &job) {
  try {
    // Do the job
    if (job.job_type == ThreadPoolJobType::STEP) {
      do_job_step(job);
    } else if (job.job_type == ThreadPoolJobType::ENCODE) {
      do_job_encode(job);
    } else {
      JCHECK(false, "ThreadPoolJobType Not Implemented");
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Worker thread exception: " << e.what();
    throw e;
  }
}

void ThreadPool::do_job_step(ThreadPoolJob &job) {
  for (Game *game : job.games) {
    game->process();
    game->get_all_possible_orders();
  }
}

void ThreadPool::do_job_encode(ThreadPoolJob &job) {
  JCHECK(job.job_type == ThreadPoolJobType::ENCODE,
         "do_job_encode called with wrong ThreadPoolJobType");
  JCHECK(job.games.size() == job.encoding_array_pointers.size(),
         "do_job_encode called with wrong input sizes");

  for (int i = 0; i < job.games.size(); ++i) {
    Game *game = job.games[i];
    EncodingArrayPointers &pointers = job.encoding_array_pointers[i];

    // encode x_board_state
    encode_board_state(game->get_state(), pointers.x_board_state);

    // encode x_prev_state, x_prev_orders
    GameState *prev_move_state = game->get_last_movement_phase();
    if (prev_move_state != nullptr) {
      encode_board_state(*prev_move_state, pointers.x_prev_state);
      orders_encoder_.encode_prev_orders_deepmind(game, pointers.x_prev_orders);
    } else {
      memset(pointers.x_prev_state, 0,
             81 * BOARD_STATE_ENC_WIDTH * sizeof(float));
      memset(pointers.x_prev_orders, 0,
             2 * PREV_ORDERS_CAPACITY * sizeof(long));
    }

    // encode x_season
    Phase current_phase = game->get_state().get_phase();
    memset(pointers.x_season, 0, 3 * sizeof(float));
    if (current_phase.season == 'S') {
      pointers.x_season[0] = 1;
    } else if (current_phase.season == 'F') {
      pointers.x_season[1] = 1;
    } else {
      pointers.x_season[2] = 1;
    }

    // encode x_in_adj_phase, x_build_numbers
    if (current_phase.phase_type == 'A') {
      *pointers.x_in_adj_phase = 1;
      float *p = pointers.x_build_numbers;
      for (Power power : POWERS) {
        *p = game->get_state().get_n_builds(power);
        ++p;
      }
    } else {
      *pointers.x_in_adj_phase = 0;
      memset(pointers.x_build_numbers, 0, 7 * sizeof(float));
    }

    // encode x_possible_actions, x_loc_idxs, x_max_seq_len
    int max_seq_len = -1;
    for (int power_i = 0; power_i < 7; ++power_i) {
      int seq_len = orders_encoder_.encode_valid_orders(
          POWERS[power_i], game->get_state(),
          pointers.x_possible_actions + (power_i * orders_encoder_.MAX_SEQ_LEN *
                                         orders_encoder_.get_max_cands()),
          pointers.x_loc_idxs + (power_i * 81));

      if (seq_len > max_seq_len) {
        max_seq_len = seq_len;
      }
    }
    *pointers.x_max_seq_len = max_seq_len;
  }
}

} // namespace dipcc
