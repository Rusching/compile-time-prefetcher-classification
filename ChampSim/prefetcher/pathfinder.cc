#include "pathfinder.h"
#include "champsim.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

/* SNN library includes for Network class. */
#include "network.hpp"
#include "receptive_field.hpp"
#include "utils.hpp"

/* Includes being really weird (like won't event read std::reduce(), so just
 * redeclaring here. */
template <class InputIterator>
constexpr typename std::iterator_traits<InputIterator>::value_type reduce(
    InputIterator first, InputIterator last)
{
    using ValueType = typename std::iterator_traits<InputIterator>::value_type;
    return std::accumulate(first, last, ValueType{});
}

template <typename T>
T interpolate(const std::vector<T> &xData, const std::vector<T> &yData, T x,
              bool extrapolate = false)
{
    int size = xData.size();

    // find left end of interval for interpolation
    int i = 0;

    // special case: beyond right end
    if (x >= xData[size - 2])
    {
        i = size - 2;
    }
    else

    {
        while (x > xData[i + 1])
            i++;
    }

    // points on either side (unless beyond ends)
    T xL = xData[i], yL = yData[i], xR = xData[i + 1], yR = yData[i + 1];
    // if beyond ends of array and not extrapolatin
    if (!extrapolate)
    {
        if (x < xL)
            yR = yL;
        if (x > xR)
            yL = yR;
    }

    // gradient
    T dydx = (yR - yL) / (xR - xL);
    // linear interpolation
    return yL + dydx * (x - xL);
}

template <typename MapType> void evict_least_used(MapType &container)
{
    uint64_t test_evict, evict_idx = 0, evict_num = ~static_cast<uint64_t>(0);

    for (const auto &pair : container)
    {
        test_evict = pair.second.get()->evict; // Access the `evict` member
        if (test_evict < evict_num)
        {
            evict_idx = pair.first; // Track the key of the least-used item
            evict_num = test_evict;
        }
    }

    container.erase(evict_idx); // Remove the entry with the smallest `evict`
}

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
    last_addr = 0;
    last_pred = 0;
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

void PathfinderPrefetcher::update_pixel_matrix(int *delta_pattern)
{
    const int offset_adj = net.inp_x / 2;
    int offset;

    /* Set every row to 0. */
    mat.setTo(0);

    /* First row. */
    offset = delta_pattern[0] + offset_adj;
    mat.at<int>(0, offset) = knob::pf_input_intensity;
    mat.at<int>(0, (offset - 1) % net.inp_x) = knob::pf_input_intensity;
    mat.at<int>(0, (offset + 1) % net.inp_x) = knob::pf_input_intensity;
    mat.at<int>(1 % net.inp_y, offset) = knob::pf_input_intensity;
    mat.at<int>(net.inp_y - 1, offset) = knob::pf_input_intensity;

    /* Middle rows. */
    int prime = knob::pf_middle_offset;
    for (int j = 1; j < net.inp_y - 1; j++)
    {

        offset = (delta_pattern[j] + prime + offset_adj % net.inp_x);

        mat.at<int>(j, offset) = knob::pf_input_intensity;
        mat.at<int>(0, (offset - 1) % net.inp_x) = knob::pf_input_intensity;
        mat.at<int>(0, (offset + 1) % net.inp_x) = knob::pf_input_intensity;
        mat.at<int>(j + 1, offset) = knob::pf_input_intensity;
        mat.at<int>(j - 1, offset) = knob::pf_input_intensity;

        prime = (prime + knob::pf_middle_offset) % net.inp_x;
    }

    /* Last row. */
    offset = delta_pattern[net.inp_y - 1] + offset_adj;
    mat.at<int>(0, offset) = knob::pf_input_intensity;
    mat.at<int>(0, (offset - 1) % net.inp_x) = knob::pf_input_intensity;
    mat.at<int>(0, (offset + 1) % net.inp_x) = knob::pf_input_intensity;
    mat.at<int>((net.inp_y + 1) % net.inp_y, offset) = knob::pf_input_intensity;
    mat.at<int>(net.inp_y - 1, offset) = knob::pf_input_intensity;
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
                potential[i][j] = static_cast<float>(mat.at<int>(i, j));
            }
        }

        cout << "potential arr cast success" << endl;

        spikes_per_neuron = custom_train_on_potential(potential);

        cout << "train on potential success" << endl;
    }
    return spikes_per_neuron;
}

void PathfinderPrefetcher::invoke_prefetcher(uint64_t pc, uint64_t address,
                                             uint8_t cache_hit, uint8_t type,
                                             vector<uint64_t> &pref_addr)
{
    cout << "Invoked!" << endl;

    if (iteration++ % iter_freq == 0)
        cout << "On iteration " << iteration - 1 << endl;

    /* Calculate memory access values. */
    uint64_t page_offset = (address >> cache_block_bits) & block_mask;
    uint64_t page = (address >> page_bits) << page_bits;

    /* Increment / decrement confidence of previous predictions. */
    uint64_t last_page_offset = ((last_addr >> cache_block_bits) & block_mask)
                                << cache_block_bits;
    uint64_t last_page = (last_addr >> page_bits) << page_bits;

    cout << "Pass assignments" << endl;

    prediction_table_info_t *pt;
    for (int i = 0; i < knob::pf_max_degree; i++)
    {
        pt = &(prediction_table[last_pred][i]);

        /* Ignore empty entries. */
        if (!pt->valid)
            continue;

        /* Strengthen confidence. */
        if (page == last_page && page_offset - last_page_offset == pt->delta)
        {
            pt->confidence += 1;
            if (pt->confidence > knob::pf_max_confidence)
                pt->confidence = knob::pf_max_confidence;
        }
        /* Weaken confidence or free neuron to be assigned to new delta. */
        else
        {
            pt->confidence -= 1;
            if (pt->confidence < knob::pf_min_confidence)
                pt->valid = false;
        }
    }

    cout << "Survive prediction verification update" << endl;

    int delta;
    training_table_info_t *tt;

    /* If PC + page in training table, extract its deltas, calculate the new
     * delta, save it. */
    if (pc_in_training_table(pc) &&
        page_in_training_table(training_table[pc].get(), page))
    {
        tt = training_table.at(pc).get()->page.at(page).get();
        delta = page_offset - tt->last_offset;

        for (int i = 1; i < net.inp_y; i++)
        {
            tt->delta_pattern[i - 1] = tt->delta_pattern[i];
        }
        tt->delta_pattern[net.inp_y - 1] = delta;
    }
    /* If not in training table, create an entry, create the offset entry, save
     * entry.*/
    else
    {
        cout << "About to insert page" << endl;

        tt = insert_page(pc, page);
        delta = page_offset;

        tt->delta_pattern[0] = page_offset;
    }

    cout << "Survive TT creations" << endl;

    /* Prep pattern. */
    update_pixel_matrix(tt->delta_pattern);

    cout << "Survive pixel matrix" << endl;

    /* Feed into neural network. */
    vector<int> spikes = custom_train(1);

    cout << "Survive training" << endl;

    /* Make prediction. Just get the maximum firing neuron index for now. */
    auto max_it = std::max_element(spikes.begin(), spikes.end());
    int max_index = std::distance(spikes.begin(), max_it);

    /* Add predictions. */
    bool delta_repped = false;
    for (int i = 0; i < knob::pf_max_degree; i++)
    {
        pt = &(prediction_table[max_index][i]);

        /* Track if current delta is represented. */
        delta_repped = delta_repped || (pt->valid && (pt->delta == delta));

        pref_addr.push_back(page + (pt->delta << cache_block_bits));
    }

    cout << "Survive predicting" << endl;

    /* If there is an open space and this delta is not represented, claim it for
     * this delta. */
    if (!delta_repped)
    {
        for (int i = 0; i < knob::pf_max_degree; i++)
        {
            pt = &(prediction_table[max_index][i]);

            /* Add delta to only invalid entries. */
            if (pt->valid)
                continue;

            pt->confidence = 0;
            pt->valid = true;
            pt->delta = delta;
            break;
        }
    }

    cout << "Survive PT update" << endl;

    /* Cleanup. */
    tt->last_offset = page_offset;
    last_addr = address;
    last_pred = max_index;

    cout << "Survive cleanup" << endl;
}

void PathfinderPrefetcher::dump_stats()
{
}

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
    float thresh = custom_threshold(spike_train);
    // std::cout << "This is line number: " << __LINE__ << std::endl;

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
        // Iterate through all the output neurons
        for (int j = 0; j < this->net.layer2.size(); j++)
        {
            Neuron *neuron = &this->net.layer2.at(j);

            if (neuron->t_reset < t)
            {
                // Simulate potential accumulation from input
                vector<float> sliced = slice_col(t, spike_train);
                neuron->p +=
                    dot_sse(net.synapse[j], sliced.data(), sliced.size());

                // Adjust the potential toward resting state based on tau_plus
                // or tau_minus (done in training)
                // if (neuron->p > this->net.p_rest)
                // {
                //     // Exponential decay toward p_rest with tau_plus
                //     neuron->p -=
                //         (neuron->p - this->net.p_rest) * (1 / net.tau_plus);
                // }
                // else
                // {
                //     // Exponential decay toward p_rest with tau_minus
                //     neuron->p +=
                //         (this->net.p_rest - neuron->p) * (1 / net.tau_minus);
                // }

                // Update active potential tracking
                active_potential[j] = neuron->p;
            }
        }

        // Perform lateral inhibition if not already done
        if (!lateral_inhibition_finished)
        {
            int winner = this->net.lateral_inhibition(active_potential, thresh);
            if (winner > -1)
            {
                img_win = winner;
                lateral_inhibition_finished = true;
            }
        }

        // Count spikes and update weights
        for (int n_indx = 0; n_indx < this->net.out_dim; n_indx++)
            if (this->net.layer2.at(n_indx).check())
                num_spikes[n_indx]++;

        custom_update_weights(spike_train, t);
    }

    // std::cout << "This is line number: " << __LINE__ << std::endl;

    // if there was a winner neuron
    if (img_win > -1)
    {
        // std::cout << "This is line number: " << __LINE__ << std::endl;
        for (int p = 0; p < this->net.inp_dim; p++)
        {
            // if there were no spikes at all from this particular input neuron
            if (reduce(spike_train.at(p).begin(), spike_train.at(p).end()) == 0)
            {
                // decrease the synaptic weights
                net.synapse[img_win][p] -= this->net.a_minus * this->net.scale;
                // make sure that the weight doesn't go below minimum weight
                if (net.synapse[img_win][p] < this->net.w_min)
                {
                    net.synapse[img_win][p] = this->net.w_min;
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
                1.0,
                static_cast<float>(this->net.max_time) / this->net.time_back +
                    static_cast<float>(this->net.max_time) /
                        this->net.time_forwards};
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

void PathfinderPrefetcher::custom_update_weights(
    vector<vector<float>> &spike_train, int t)
{
    // iterate through all the neurons in the output layers
    for (int j = 0; j < net.layer2.size(); j++)
    {
        Neuron *neuron = &net.layer2.at(j);
        // check whether the neuron's potential has hit the threshold
        if (neuron->check())
        {
            // make sure that the neuron doesn't fire until it has finished
            // waiting until the refractory period
            neuron->t_reset = t + neuron->t_refractory;
            // reset potential to the resting potential
            neuron->p = net.p_rest;

            // loop over all weights
            for (int h = 0; h < this->net.inp_dim; h++)
            {
                for (int t1 = -2; t1 < this->net.time_back; t--)
                {
                    // if the look back is within the bounds of the simulation
                    if (t + t1 <= this->net.max_time && t + t1 >= 0)
                    {
                        // if it sppiked within the time bounds, then it means
                        // that the spike from this synapse probably contributed
                        // to this neuron, so update the weights
                        if (spike_train.at(h).at(t + t1) == 1)
                            net.synapse[j][h] = custom_stdp_update(
                                net.synapse[j][h],
                                custom_reinforcement_learning(t1));
                    }
                }

                // do the same thing, except for the forward times
                for (int t1 = 2; t1 < this->net.time_forwards; t1++)
                {
                    if (t + t1 >= 0 && t + t1 <= this->net.max_time)
                    {
                        // we want to decrease influence for these ones now
                        if (spike_train.at(h).at(t + t1) == 1)
                            net.synapse[j][h] = custom_stdp_update(
                                net.synapse[j][h],
                                custom_reinforcement_learning(t1));
                    }
                }
            }
        }
    }
}

float PathfinderPrefetcher::custom_stdp_update(float w, float delta_w)
{
    const float sigma = 1.0;
    if (delta_w < 0)
    {
        return (w + sigma * delta_w * (w - abs(net.w_min)) * net.scale);
    }

    return (w + sigma * delta_w * (net.w_max - w) * net.scale);
}

float PathfinderPrefetcher::custom_reinforcement_learning(int time)
{
    if (time > 0)
    {
        return -net.a_plus * exp(-(float)(time) / net.tau_plus);
    }

    return net.a_minus * exp((float)(time) / net.tau_minus);
}

float PathfinderPrefetcher::custom_threshold(vector<vector<float>> &train)
{
    int train_len = train.at(0).size();
    int thresh = 0;

    for (int i = 0; i < train_len; i++)
    {
        int col_sum = 0;
        for (int j = 0; j < train.size(); j++)
            col_sum += train.at(j).at(i);
        if (col_sum > thresh)
            thresh = col_sum;
    }

    return (thresh / 3) * net.scale;
}

training_table_info_t *PathfinderPrefetcher::insert_page(uint64_t pc,
                                                         uint64_t page)
{
    lru_pc_t *pt;
    training_table_info_t *tt;

    if (training_table.size() >= training_table_pc_len)
        evict_least_used(training_table);

    unique_ptr<lru_pc_t> new_page = std::make_unique<lru_pc_t>();
    training_table[pc] = std::move(new_page);

    pt = training_table[pc].get();
    pt->evict = lru_evict++;

    cout << "Survive pc entry assignment" << endl;

    if (pt->page.size() >= training_table_page_len)
        evict_least_used(pt->page);

    unique_ptr<training_table_info_t> net_tt =
        std::make_unique<training_table_info_t>();
    pt->page[page] = std::move(net_tt);

    tt = pt->page[page].get();

    cout << "Survive page entry assignment" << endl;

    /* Init training table info. */
    tt->evict = lru_evict++;
    tt->last_offset = 0;
    for (int i = 0; i < net.inp_y; i++)
    {
        tt->delta_pattern[i] = 0;
    }

    return tt;
}

bool PathfinderPrefetcher::pc_in_training_table(uint64_t pc)
{
    bool out = training_table.find(pc) != training_table.end();

    if (out)
        training_table[pc]->evict = lru_evict++;

    return out;
}

bool PathfinderPrefetcher::page_in_training_table(lru_pc_t *pt, uint64_t page)
{
    bool out = pt->page.find(page) != pt->page.end();

    if (out)
        pt->page[page].get()->evict = lru_evict++;

    return out;
}
