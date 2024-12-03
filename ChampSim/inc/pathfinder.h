#ifndef PATHFINDER_H
#define PATHFINDER_H

#include <vector>
#include "prefetcher.h"
#include "network.hpp"

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

private:
    void init_knobs();
    void init_stats();
    void update_pixel_matrix(uint64_t address, bool page_change, int offset_idx);
    vector<int> custom_train(const int epochs);

public:
    PathfinderPrefetcher(string type);
    ~PathfinderPrefetcher();
    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t> &pref_addr);
    void dump_stats();
    void print_config();
};

#endif /* PATHFINDER_H */
