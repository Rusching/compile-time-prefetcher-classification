#ifndef PATHFINDER_H
#define PATHFINDER_H

#include "../snnpp/include/network.hpp"
#include "prefetcher.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

using namespace std;

namespace knob
{
extern uint32_t pf_pattern_len;
extern int32_t pf_confidence_threshold;
extern int32_t pf_min_confidence;
extern uint32_t pf_delta_range_len;
extern uint32_t pf_neuron_numbers;
extern uint32_t pf_timestamps;
extern uint32_t pf_input_intensity;
uint32_t pf_max_confidence = 5; // Really dumb that they forgot to make this a
                                // constant. It's a 3-bit saturating counter.
int32_t pf_max_degree = 2;
uint32_t pf_middle_offset =
    67; // Prevent aliasing in neuron input. Should be prime.
} // namespace knob

typedef struct lru_pc
{
    unordered_map<uint64_t, unique_ptr<training_table_info_t>> page;
    uint64_t evict;
} lru_pc_t;

typedef struct training_table_info
{
    int fired_neuron;
    int last_offset;
    int delta_pattern[knob::pf_pattern_len]; // Use smart pointers for arrays
    uint64_t evict;
} training_table_info_t;

typedef struct prediction_table_info
{
    int delta;
    int confidence;
    bool valid = false;
} prediction_table_info_t;

class PathfinderPrefetcher : public Prefetcher
{
  private:
    Network net;

    /* Training tables. */
    unordered_map<uint64_t, unique_ptr<lru_pc_t>> training_table;
    prediction_table_info_t prediction_table[knob::pf_neuron_numbers]
                                            [knob::pf_max_degree];

    cv::Mat mat;

    int page_bits;
    int cache_block_bits;

    uint64_t block_mask;
    uint64_t page_mask;

    uint64_t last_addr;
    int last_pred;

    uint64_t iteration;
    const int iter_freq = 50000;

    /* Training table variables. */
    uint64_t LRU_evict = 0;
    const int training_table_PC_len;
    const int training_table_page_len;

  private:
    void init_knobs();
    void init_stats();
    void update_pixel_matrix(int new_delta, int *old_delta_pattern);
    vector<int> custom_train(const int epochs);
    vector<int> custom_train_on_potential(vector<vector<float>> &potential);
    vector<vector<float>> poisson_encode(vector<vector<float>> &potential);
    void custom_update_weights(vector<vector<float>> &spike_train, int t);
    float custom_stdp_update(float w, float delta_w);
    float custom_reinforcement_learning(int time);
    void add_predictions_to_queue(uint64_t pc, uint64_t page,
                                  uint64_t page_offset,
                                  vector<uint64_t> &pref_addr);
    void update_training_table(uint64_t pc, uint64_t page, uint64_t page_offset,
                               fired_neurons, new_delta_pattern,
                               bool is_page_offset);
    void make_prediction();

  public:
    PathfinderPrefetcher(string type);
    ~PathfinderPrefetcher();
    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit,
                           uint8_t type, vector<uint64_t> &pref_addr);
    void dump_stats();
    void print_config();
};

#endif /* PATHFINDER_H */
