#include "pathfinder.h"
#include "champsim.h"
#include "network.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

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
} // namespace knob

/* Constructor. */
PathfinderPrefetcher::PathfinderPrefetcher(string type)
    : Prefetcher(type),
      net(knob::pf_delta_range_len,        // inp_x: delta_range_length
          knob::pf_pattern_len,            // inp_y: pattern_length
          knob::pf_neuron_numbers,         // out_dim: neuron_numbers
          knob::pf_delta_range_len / 10.0, // scale: norm
          knob::pf_timestamps,             // max_time: timestamps

          /* All these constants synthesized from the DiehlAndCook2015 Python
             class default constructor + Pythia pathfinder implementation. */
          1,     // time_back: dt (fixed as 1)
          1,     // time_forwards: dt (fixed as 1)
          -40.0, // p_min: inh_thresh
          -52.0, // p_rest: exc_thresh
          1.0,   // w_max: wmax
          0.0,   // w_min: wmin
          0.05,  // a_plus: theta_plus
          0.01,  // a_minus: small decrement for STDP
          1e7,   // tau_plus: tc_theta_decay
          1e7,   // tau_minus: assumed same as tau_plus
          true   // simd_optimized: enable optimizations
      )
{
    // Memory allocation and bitmask calculation
    offsets = (u_char *)calloc(knob::pf_pattern_len, sizeof(u_char));
    last_addr = 0;
    page_bits = log2(PAGE_SIZE);
    cache_block_bits = log2(BLOCK_SIZE);

    block_mask = (1 << (page_bits - cache_block_bits)) - 1;
    page_mask = (1 << page_bits) - 1;

    // Debugging output
    cout << knob::pf_pattern_len << "x" << knob::pf_delta_range_len << " matrix"
         << endl;

    // OpenCV matrix initialization
    mat = cv::Mat(knob::pf_pattern_len, knob::pf_delta_range_len, CV_8UC1,
                  cv::Scalar(0));

    /* Fix neuron parameters. */
    for (Neuron &n : net.layer2)
    {
        n.p_min = net.p_min;
        n.p_rest = net.p_rest;
        n.t_refractory = 0;
    }
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
    cout << "Pathfinder configuration: " << knob::pf_pattern_len << " row x "
         << knob::pf_delta_range_len << " col matrix" << endl;
}

void PathfinderPrefetcher::update_pixel_matrix(uint64_t address,
                                               bool page_change, int offset_idx)
{
    int offset_down = max(offset_idx - 1, 0);
    int offset_up = min(offset_idx + 1, this->net.inp_x - 1);

    /* Set up pixel matrix special case for a new page. */
    if (page_change)
    {
        cout << "trying page change" << endl;

        /* Set every row to 0. */
        mat.setTo(0);

        /* Set the first row, column determined by address, to 1. */
        mat.at<u_char>(0, offset_down) = knob::pf_input_intensity;
        mat.at<u_char>(0, offset_idx) = knob::pf_input_intensity;
        mat.at<u_char>(0, offset_up) = knob::pf_input_intensity;
        return;
    }

    cout << "trying established history prefetch" << endl;

    /* Move rows up by one. */
    for (int i = 0; i < this->net.inp_y - 1; i++)
    {
        mat.row(i) = mat.row(i + 1);
    }
    mat.row(this->net.inp_y - 1) = cv::Scalar(0);
    mat.at<u_char>(this->net.inp_y - 1, offset_down) = knob::pf_input_intensity;
    mat.at<u_char>(this->net.inp_y - 1, offset_idx) = knob::pf_input_intensity;
    mat.at<u_char>(this->net.inp_y - 1, offset_up) = knob::pf_input_intensity;
}

vector<int> PathfinderPrefetcher::custom_train(const int epochs)
{
    vector<int> spikes_per_neuron;
    for (int k = 0; k < epochs; k++)
    {
        vector<vector<float>> potential(mat.rows, vector<float>(mat.cols, 0));

        cout << "potential success" << endl;

        /* Convert Mat to vector. This whole library should really be rewritten
         * but time. */
        for (int i = 0; i < mat.rows; i++)
        {
            for (int j = 0; j < mat.cols; j++)
            {
                potential[i][j] = static_cast<float>(mat.at<u_char>(i, j));
            }
        }

        cout << "potential arr cast success" << endl;

        spikes_per_neuron = net.train_on_potential(potential);

        cout << "train on potential success" << endl;
    }
    return spikes_per_neuron;
}

void PathfinderPrefetcher::invoke_prefetcher(uint64_t pc, uint64_t address,
                                             uint8_t cache_hit, uint8_t type,
                                             vector<uint64_t> &pref_addr)
{
    // cout << "Invoked!" << endl;

    if (iteration++ % iter_freq == 0)
        cout << "On iteration " << iteration - 1 << endl;

    /* Update input pixel matrix. */
    bool page_change = !((last_addr & page_mask) == (address & page_mask));
    int offset =
        (static_cast<int64_t>(address) - static_cast<int64_t>(last_addr)) &
        block_mask;
    int offset_idx = offset + this->net.inp_x / 2;

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

/*
 * train_on_potential
 * adjust the weights of the network for one simulation run (0-200 time units)
 * based on one single input image (which must be preprocessed into membrane
 * potential)
 *
 * @param potential
 * @return the number of spikes of every neuron
 */
vector<int> PathfinderPrefetcher::custom_train_on_potential(
    vector<vector<float>> &potential)
{
    vector<int> num_spikes(this->net.out_dim);

    // encode the potential produced by running it through the receptive field
    // into a spike train so that it can be given as input to the network
    vector<vector<float>> spike_train = poisson_encode(potential);
    // vector<vector<float>> spike_train = potential;
    // std::cout << "This is line number: " << __LINE__ << std::endl;

    // get a custom threshold for the given image
    float thresh = threshold(spike_train);
    // std::cout << "This is line number: " << __LINE__ << std::endl;

    float var_D = 0.15 * this->net.scale;

    for (int i = 0; i < this->net.layer2.size(); i++)
        this->net.layer2.at(i).initial(thresh);

    // std::cout << "This is line number: " << __LINE__ << std::endl;

    // we only want to do lateral inhibition once per time
    bool lateral_inhibition_finished = false;
    int img_win = -1;

    vector<int> active_potential(this->net.out_dim, 0);
    // std::cout << "This is line number: " << __LINE__ << std::endl;

    // simulate the network through the entire time period
    for (int t = 1; t <= this->net.max_time; t++)
    {
        // iterate through all the output neurons
        for (int j = 0; j < this->net.layer2.size(); j++)
        {
            Neuron *neuron = &this->net.layer2.at(j);
            // std::cout << "This is line number: " << __LINE__ << std::endl;

            if (neuron->t_reset < t)
            {
                // simulate the potential accumulation in the neuron
                // by doing a dot product of the input spike train with the
                // synaptic weights for the current neuron
                vector<float> sliced = slice_col(t, spike_train);
                // std::cout << "This is line number: " << __LINE__ <<
                // std::endl;

                neuron->p += dot_sse(synapse[j], sliced.data(), sliced.size());
                // std::cout << "This is line number: " << __LINE__ <<
                // std::endl;

                // if the potential is greater than resting potential
                if (neuron->p > this->net.p_rest)
                {
                    // decrease the potential
                    neuron->p -= var_D;
                }
                active_potential[j] = neuron->p;
            }
            // std::cout << "This is line number: " << __LINE__ << std::endl;
        }

        // perform lateral inhibition if it has not been performed already
        if (!lateral_inhibition_finished)
        {
            int winner = this->net.lateral_inhibition(active_potential, thresh);
            if (winner != -1)
            {
                img_win = winner;
                lateral_inhibition_finished = true;
            }
        }
        // std::cout << "This is line number: " << __LINE__ << std::endl;

        // for every neuron in the output layer, note how many times each neuron
        // spiked up to this point in time
        for (int n_indx = 0; n_indx < this->net.out_dim; n_indx++)
            if (this->net.layer2.at(n_indx).check())
                num_spikes[n_indx]++;

        // std::cout << "This is line number: " << __LINE__ << std::endl;

        // check for spikes and update weights accordingly
        this->net.update_weights(spike_train, t);
        // std::cout << "This is line number: " << __LINE__ << std::endl;
    }
    // std::cout << "This is line number: " << __LINE__ << std::endl;

    // if there was a winner neuron
    if (img_win != -1)
    {
        // std::cout << "This is line number: " << __LINE__ << std::endl;
        for (int p = 0; p < this->net.inp_dim; p++)
        {
            // if there were no spikes at all from this particular input neuron
            if (reduce(spike_train.at(p).begin(), spike_train.at(p).end()) == 0)
            {
                // decrease the synaptic weights
                synapse[img_win][p] -= this->net.a_minus * this->net.scale;
                // make sure that the weight doesn't go below minimum weight
                if (synapse[img_win][p] < this->net.w_min)
                {
                    synapse[img_win][p] = this->net.w_min;
                }
            }
        }
    }

    // std::cout << "This is line number: " << __LINE__ << std::endl;

    // return the total number of spikes for each neuron
    return num_spikes;
}

/*
 * encode
 * given a 2d membrane potential, this function encodes it into
 * a spike train that can be fed directly into the SNN as input
 */
vector<vector<float>> PathfinderPrefetcher::poisson_encode(
    vector<vector<float>> &potential)
{
    // dims will be such that train.size() = input dimensions and
    // train[i].size() = max simulation time
    vector<vector<float>> train;

    vector<vector<float>> train;

    // Poisson distribution parameters
    std::random_device rd;
    std::mt19937 gen(rd());

    // iterate through all the pixels
    for (int i = 0; i < potential.size(); i++)
    {
        for (int j = 0; j < potential.at(0).size(); j++)
        {
            vector<float> temp(this->net.max_time + 1, 0.0);

            // convert the membrane potential values into
            // spikes range that can be fed into the network
            const vector<float> r1 = {this->net.w_min, this->net.w_max};
            const vector<float> r2 = {
                1.0, this->net.max_time / this->net.time_back +
                         this->net.max_time / this->net.time_forwards};
            float freq = interpolate(r1, r2, potential.at(i).at(j));

            if (freq <= 0)
            {
                // std::cerr << "Frequency is out of range" << std::endl;
                continue; // Skip this pixel if frequency is invalid
            }

            // Use Poisson distribution to determine the spike timings
            std::poisson_distribution<> poisson(freq);

            for (int t = 1; t <= this->net.max_time; ++t)
            {
                // Generate the number of spikes that would occur at time t
                int spikes_at_t = poisson(gen);
                if (spikes_at_t > 0)
                {
                    temp[t] = 1.0; // Mark a spike at this time step
                }
            }

            train.push_back(temp);
        }
    }

    return train;
}