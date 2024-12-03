#include <algorithm>
#include <cmath>
#include <iostream>
#include "pathfinder.h"
#include "champsim.h"

#include "receptive_field.hpp"

namespace knob
{
    extern uint32_t pf_pattern_len;
    extern int32_t pf_confidence_threshold;
    extern int32_t pf_min_confidence;
    extern uint32_t pf_delta_range_len;
    extern uint32_t pf_neuron_numbers;
    extern uint32_t pf_timestamps;
    extern uint32_t pf_input_intensity;
}

// Constructor
PathfinderPrefetcher::PathfinderPrefetcher(string type)
    : Prefetcher(type),
      net(1, knob::pf_delta_range_len * knob::pf_pattern_len, knob::pf_neuron_numbers)
{
    offsets = (u_char *)calloc(knob::pf_pattern_len, sizeof(u_char));
    last_addr = 0;
    page_bits = log2(PAGE_SIZE);
    cache_block_bits = log2(BLOCK_SIZE);

    block_mask = (1 << (page_bits - cache_block_bits)) - 1;
    page_mask = (1 << page_bits) - 1;

    mat = cv::Mat(knob::pf_pattern_len, knob::pf_delta_range_len, CV_8UC1, cv::Scalar(0));
    // mat.at<uchar>(2, 3) = 255;
}

// Destructor
PathfinderPrefetcher::~PathfinderPrefetcher()
{
    free(offsets);
}

void PathfinderPrefetcher::init_knobs()
{
}

void PathfinderPrefetcher::init_stats()
{
}

void PathfinderPrefetcher::print_config()
{
    cout << "TODO: implement Pathfinder config";
}

void PathfinderPrefetcher::update_pixel_matrix(uint64_t address, bool page_change, int offset_idx)
{
    int offset_down = std::max(offset_idx - 1, 0);
    int offset_up = std::min(offset_idx + 1, static_cast<int>(knob::pf_delta_range_len - 1));

    /* Set up pixel matrix special case for a new page. */
    if (page_change)
    {
        /* Set every row to 0. */
        mat.setTo(0);

        /* Set the first row, column determined by address, to 1. */
        mat.at<u_char>(0, offset_down) = 1;
        mat.at<u_char>(0, offset_idx) = 1;
        mat.at<u_char>(0, offset_up) = 1;
        return;
    }

    /* Move rows up by one. */
    for (int i = 0; i < knob::pf_pattern_len - 1; i++)
    {
        mat.row(i) = mat.row(i + 1);
    }
    mat.row(knob::pf_pattern_len - 1) = cv::Scalar(0);
    mat.at<u_char>(knob::pf_pattern_len - 1, offset_down) = 1;
    mat.at<u_char>(knob::pf_pattern_len - 1, offset_idx) = 1;
    mat.at<u_char>(knob::pf_pattern_len - 1, offset_up) = 1;
}

vector<int> PathfinderPrefetcher::custom_train(const int epochs)
{
    vector<int> spikes_per_neuron;
    for (int k = 0; k < epochs; k++)
    {
        vector<vector<float>> potential(mat.rows, vector<float>(mat.cols, 0));

        /* Convert Mat to vector. This whole library should really be rewritten
         * but time. */
        for (int i = 0; i < mat.rows; i++)
        {
            for (int j = 0; j < mat.cols; j++)
            {
                potential[i][j] = static_cast<float>(mat.at<u_char>(i, j));
            }
        }

        spikes_per_neuron = net.train_on_potential(potential);
    }
    return spikes_per_neuron;
}

void PathfinderPrefetcher::invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t> &pref_addr)
{
    /* Update input pixel matrix. */
    bool page_change = !(last_addr & page_mask == address & page_mask);
    int offset = (static_cast<int64_t>(address) - static_cast<int64_t>(last_addr)) & block_mask;
    int offset_idx = offset + knob::pf_delta_range_len / 2;

    update_pixel_matrix(address, page_change, offset_idx);

    /* Feed into neural network. */
    vector<int> spikes = custom_train(1);

    /* Update prediction table. */

    /* Complete prefetch. */
    pref_addr.push_back(address);
    last_addr = address;
}

void PathfinderPrefetcher::dump_stats()
{
}
