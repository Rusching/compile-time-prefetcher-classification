#include "common.h"

#include <cmath>

int transfer(int origin) {
    return std::abs(origin) * origin; 
}

int 
count_bits(uint64_t a) 
{
    int res = 0;
    for (int i = 0; i < 64; i++)
        if ((a >> i) & 1)
            res++;
    return res;
}

uint64_t 
pattern_to_int(const vector<bool> &pattern)
{
	uint64_t result = 0;
	for (auto v : pattern)
	{
		result <<= 1;
		result |= int(v);
	}
	return result;
}

vector<bool> 
pattern_convert2(const vector<int> &x) 
{
    vector<bool> pattern;
    for (int i = 0; i < x.size(); i++) {
        pattern.push_back(x[i] != 0);
    }
    return pattern;
}

vector<bool> 
pattern_convert2(const vector<uint32_t> &x) 
{
    vector<bool> pattern;
    for (int i = 0; i < x.size(); i++) {
        pattern.push_back(x[i] != 0);
    }
    return pattern;
}

vector<int> 
pattern_convert(const vector<bool> &x) {
    vector<int> pattern(x.size(), 0);
    for (int i = 0; i < x.size(); i++) {
        pattern[i] = (x[i] ? 1 : 0);
    }
    return pattern;
}

int 
count_bits(const vector<bool> &x) 
{
    int count = 0;
    for (auto b: x) {
        count += b;
    } 
    return count;
}

double 
jaccard_similarity(vector<bool> pattern1, vector<bool> pattern2) 
{
    int a = pattern_to_int(pattern1), b = pattern_to_int(pattern2);
    return double(count_bits(a & b)) / double(count_bits(a | b));
}

double 
jaccard_similarity(vector<bool> pattern1, vector<int> pattern2) 
{
    int l = 0, j= 0;
    for (int i = 0; i < pattern1.size(); i++) {
        l += pattern1[i] ? pattern2[i] : 0;
        j += int(pattern1[i]) > pattern2[i] ? int(pattern1[i]) : pattern2[i];
    }
    return double(l) / j;
}

int pattern_distance(uint64_t p1, uint64_t p2)
{
    return count_bits(p1^p2);
}

uint32_t encode_metadata(int stride, uint16_t type, int spec_nl)
{
  uint32_t metadata = 0;

  // first encode stride in the last 8 bits of the metadata
  if (stride > 0)
    metadata = stride;
  else
    metadata = ((-1 * stride) | 0b1000000);

  // encode the type of IP in the next 4 bits
  metadata = metadata | (type << 8);

  // encode the speculative NL bit in the next 1 bit
  metadata = metadata | (spec_nl << 12);

  return metadata;
}

vector<bool> 
pattern_degrade(const vector<bool> &x, int level) 
{
    vector<bool> res(x.size()/level, false);
    for (int i = 0; i < x.size(); i++) {
        res[i/level] = res[i/level] || x[i];
    }
    return res;
}