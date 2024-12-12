#ifndef _COMMON_H_
#define _COMMON_H_

// ***************** borrow from IPCP *********************
#define S_TYPE 1    // stream
#define CS_TYPE 2   // constant stride
#define CPLX_TYPE 3 // complex stride
#define NL_TYPE 4   // next line

#define ADD(x, MAX) (x = x >= MAX ? x : x + 1)
#define ADD_ANY(x, y, MAX) (x = x + y >= MAX ? MAX : x + y)
#define TIME(x, y, MAX) (x = x * y > MAX ? MAX : x * y)
#define SHIFT(x, y, MAX) (x = (x << y) > MAX ? MAX : x << y)
#define ADD_BACKOFF(x, MAX) (x = (x == MAX ? x >> 1 : x + 1))
#define SUB(x, MIN) (x = (x <= MIN ? x : x - 1))
#define FLIP(x) (~x)

#include <stdint.h>
#include <vector>
#include <unordered_map>
#include <sstream>

using namespace std;

int transfer(int origin);
int count_bits(uint64_t a);
int count_bits(const vector<bool> &x);
uint64_t pattern_to_int(const vector<bool> &pattern);
vector<bool> pattern_convert2(const vector<int> &x);
vector<bool> pattern_convert2(const vector<uint32_t> &x);
vector<int> pattern_convert(const vector<bool> &x);
vector<bool> pattern_degrade(const vector<bool> &x, int level);

double jaccard_similarity(vector<bool> pattern1, vector<bool> pattern2);
double jaccard_similarity(vector<bool> pattern1, vector<int> pattern2);
int pattern_distance(uint64_t p1, uint64_t p2);
uint64_t hash_index(uint64_t key, int index_len);
uint32_t encode_metadata(int stride, uint16_t type, int spec_nl);

template <class T>
string pattern_to_string(const vector<T> &pattern)
{
    ostringstream oss;
    for (unsigned i = 0; i < pattern.size(); i += 1)
        oss << int(pattern[i]) << " ";
    return oss.str();
}

template <class T>
vector<T> my_rotate(const vector<T> &x, int n)
{
    vector<T> y;
    int len = x.size();
    if (len == 0)
        return y;
    n = n % len;
    for (int i = 0; i < len; i += 1)
        y.push_back(x[(i - n + len) % len]);
    return y;
}

#endif