#ifndef PATHFINDER_H
#define PATHFINDER_H

#include "../snnpp/include/network.hpp"
#include "prefetcher.h"
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;

class PathfinderPrefetcher : public Prefetcher
{
  private:
    Network net;

    cv::Mat mat;
    u_char *offsets;

    int page_bits;
    int cache_block_bits;

    uint64_t block_mask;
    uint64_t page_mask;

    uint64_t last_addr;

    uint64_t iteration;
    const int iter_freq = 50000;

  private:
    void init_knobs();
    void init_stats();
    void update_pixel_matrix(uint64_t address, bool page_change,
                             int offset_idx);
    vector<int> custom_train(const int epochs);
    vector<int> custom_train_on_potential(vector<vector<float>> &potential);
    vector<vector<float>> poisson_encode(vector<vector<float>> &potential);

  public:
    PathfinderPrefetcher(string type);
    ~PathfinderPrefetcher();
    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit,
                           uint8_t type, vector<uint64_t> &pref_addr);
    void dump_stats();
    void print_config();
};

#endif /* PATHFINDER_H */
