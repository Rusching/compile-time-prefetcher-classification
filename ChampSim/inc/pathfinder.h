#ifndef SNN_PREFETCHER_H
#define SNN_PREFETCHER_H

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "carlsim.h" // Include the CARLsim library for spiking neural networks

#include "prefetcher.h" // Include the base Prefetcher class

class SNNPrefetcher : public Prefetcher
{
private:
    CARLsim *sim;                                                      // Pointer to the CARLsim simulation object
    int inputLayer;                                                    // Input layer ID
    int excitatoryLayer;                                               // Excitatory layer ID
    int patternLength;                                                 // Length of input patterns
    int deltaRangeLength;                                              // Length of the delta range
    int neuronNumbers;                                                 // Number of neurons in the excitatory layer
    int timestamps;                                                    // Simulation timestamps
    float inputIntensity;                                              // Input intensity for encoding
    std::map<int, std::vector<std::pair<int, float>>> predictionTable; // Prediction table

    // Builds an enlarged input array for the SNN based on the delta pattern
    std::vector<int> buildEnlargedInputArray(const std::vector<int> &deltaPattern);

    // Feeds the input pattern to the SNN and gets firing neurons
    std::vector<int> feedInput(const std::vector<int> &deltaPattern);

    // Makes predictions based on the delta pattern using the SNN
    std::pair<std::vector<int>, std::vector<int>> makePrediction(const std::vector<int> &deltaPattern);

    // Updates the prediction table with neuron-delta mappings
    void updatePredictionTable(int neuron, int delta);

public:
    // Constructor to initialize the SNN prefetcher with the given parameters
    SNNPrefetcher(int pl, int drl, int nn, int ts, float ii);

    // Destructor to clean up allocated resources
    ~SNNPrefetcher();

    // Implementation of the invoke_prefetcher function to use the SNN for prefetching
    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, std::vector<uint64_t> &pref_addr) override;

    // Dumps statistics about the prefetcher
    void dump_stats() override;

    // Prints the configuration of the prefetcher
    void print_config() override;
};

#endif /* SNN_PREFETCHER_H */
