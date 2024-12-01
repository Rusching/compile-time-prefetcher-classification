#ifndef PREFETCHER_H
#define PREFETCHER_H

#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <map>
#include <utility>
#include "carlsim.h" // Include CARLsim library

class Prefetcher
{
protected:
    std::string type;

public:
    Prefetcher(std::string _type) { type = _type; }
    virtual ~Prefetcher() {}
    std::string get_type() { return type; }
    virtual void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, std::vector<uint64_t> &pref_addr) = 0;
    virtual void dump_stats() = 0;
    virtual void print_config() = 0;
};

class SNNPrefetcher : public Prefetcher
{
private:
    // SNN network
    CARLsim *sim;
    int inputLayer;
    int excitatoryLayer;
    int patternLength;
    int deltaRangeLength;
    int neuronNumbers;
    int timestamps;
    float inputIntensity;
    std::map<int, std::vector<std::pair<int, float>>> predictionTable;

    std::vector<int> buildEnlargedInputArray(const std::vector<int> &deltaPattern)
    {
        std::vector<int> validIndexList1D;

        for (int i = 0; i < patternLength; ++i)
        {
            int validIndex1D = static_cast<int>(deltaPattern[i] + (deltaRangeLength - 1) / 2);
            validIndexList1D.push_back(validIndex1D);
        }

        return validIndexList1D;
    }

    std::vector<int> feedInput(const std::vector<int> &deltaPattern)
    {
        std::vector<int> nonZeroIndices = buildEnlargedInputArray(deltaPattern);

        SpikeGeneratorFromFile sg(inputLayer);
        for (int index : nonZeroIndices)
        {
            sg.addSpike(index, 0, timestamps - 1); // Add spikes for input neurons
        }

        sim->runNetwork(0, timestamps);

        std::vector<int> firingNeurons;
        SpikeMonitor *spikeMon = sim->getSpikeMonitor(excitatoryLayer);
        spikeMon->stopRecording();
        for (int n = 0; n < neuronNumbers; ++n)
        {
            if (!spikeMon->getSpikeVector2D()[n].empty())
            {
                firingNeurons.push_back(n);
            }
        }
        sim->resetState();

        return firingNeurons;
    }

    std::pair<std::vector<int>, std::vector<int>> makePrediction(const std::vector<int> &deltaPattern)
    {
        std::vector<int> outputNeurons = feedInput(deltaPattern);
        std::vector<int> predictionDeltas;

        if (!outputNeurons.empty() && predictionTable.find(outputNeurons[0]) != predictionTable.end())
        {
            for (auto deltaTuple : predictionTable[outputNeurons[0]])
            {
                predictionDeltas.push_back(deltaTuple.first);
            }
        }
        else
        {
            predictionDeltas.clear();
        }

        return {outputNeurons, predictionDeltas};
    }

    void updatePredictionTable(int neuron, int delta)
    {
        if (predictionTable.find(neuron) == predictionTable.end())
        {
            predictionTable[neuron] = {{delta, 0.0f}};
        }
        else
        {
            bool found = false;
            for (auto &pair : predictionTable[neuron])
            {
                if (pair.first == delta)
                {
                    pair.second += 1.0f; // Increment confidence
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                predictionTable[neuron].emplace_back(delta, 0.0f);
            }
        }
    }

public:
    SNNPrefetcher(int pl, int drl, int nn, int ts, float ii)
        : Prefetcher("SNNPrefetcher"), patternLength(pl), deltaRangeLength(drl),
          neuronNumbers(nn), timestamps(ts), inputIntensity(ii)
    {
        sim = new CARLsim("SNNPrefetcher", GPU_MODE, USER);

        inputLayer = sim->createSpikeGeneratorGroup("Input", deltaRangeLength * patternLength, EXCITATORY_NEURON);
        excitatoryLayer = sim->createGroup("Excitatory", neuronNumbers, EXCITATORY_NEURON);
        sim->setNeuronParameters(excitatoryLayer, 0.02f, 0.2f, -65.0f, 8.0f); // Izhikevich params

        sim->connect(inputLayer, excitatoryLayer, "full", RangeWeight(0.1f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
        sim->setConductances(true);

        sim->setSpikeMonitor(inputLayer, "DEFAULT");
        sim->setSpikeMonitor(excitatoryLayer, "DEFAULT");
    }

    ~SNNPrefetcher() { delete sim; }

    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, std::vector<uint64_t> &pref_addr) override
    {
        std::vector<int> deltaPattern = {static_cast<int>(pc % 10), static_cast<int>(address % 10), static_cast<int>(type)};
        auto [outputNeurons, predictionDeltas] = makePrediction(deltaPattern);

        if (!predictionDeltas.empty())
        {
            for (int delta : predictionDeltas)
            {
                uint64_t predictedAddr = address + delta;
                pref_addr.push_back(predictedAddr);
            }
        }
    }

    void dump_stats() override
    {
        std::cout << "SNN Prefetcher Statistics:" << std::endl;
        std::cout << "Number of neurons: " << neuronNumbers << std::endl;
    }

    void print_config() override
    {
        std::cout << "SNN Prefetcher Configuration:" << std::endl;
        std::cout << "Pattern length: " << patternLength << std::endl;
        std::cout << "Delta range length: " << deltaRangeLength << std::endl;
        std::cout << "Timestamps: " << timestamps << std::endl;
    }
};

#endif /* PREFETCHER_H */
