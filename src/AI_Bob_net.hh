#ifndef AI_BOB_NET
#define AI_BOB_NET

#include <bits/stdc++.h>
#include <algorithm>
#include <vector>

namespace AI_BOB
{
    
class simple_network
{
public:

        template <typename... params>
    explicit simple_network(int size1, int size2, params... args) : __input_size{size1}, __output_size{std::get<sizeof...(params)-1>(std::forward_as_tuple(args...))}
     {  std::vector<int> vec = {size1,size2, ((int) args)...};
        for (int i = 1; i < vec.size(); i++)
            layers.push_back(hidden_layer(vec[i],vec[i-1]));}

    std::vector<uint8_t> get(std::vector<uint8_t>& _vec);

private:

    class hidden_layer;
    int __input_size{0};
    int __output_size{0};
    std::vector<hidden_layer> layers;

};


class simple_network::hidden_layer
{
public:
    explicit hidden_layer(const int this_size, const int prev_size): __tsize{this_size}, __psize{prev_size} 
        {  __results = new uint8_t[__tsize];
        __wb_table = new wb_pair*[__tsize]; 
        std::for_each(__wb_table, __wb_table+__tsize, [=](wb_pair* wb) mutable {wb = new wb_pair[__psize];}); }
    
    void import_values() {}

    void calculate_results(uint8_t* start)
    {  
        for (int i = 0; i < __tsize; i++)
        {
            __results[i] = 0;
            for (int j = 0; j < __psize; j++)
            {
                __results[i] += trim(apply_wb(i, j, start[j]));
            }
        }
    }

    std::vector<uint8_t> get_results()
        {  return [&]() -> std::vector<uint8_t> 
            {  std::vector<uint8_t> vec;
            for (int i = 0; i  < __tsize; i++) vec.push_back(__results[i]);
            return vec; }();    }

private:
    inline uint16_t apply_wb(int this_index, int other_index, uint8_t value)
        {  const wb_pair& wb = __wb_table[this_index][other_index];
        return ((uint16_t) value) * wb.weight + wb.bias;        }

    inline uint8_t trim(uint16_t value)
        {  return (uint8_t) (value >> 8);   }

    struct wb_pair
    {
        uint16_t weight;
        int16_t bias;
    };

    wb_pair** __wb_table{nullptr};
    uint8_t* __results{nullptr};
    const int __tsize{0};
    const int __psize{0};
};

std::vector<uint8_t> simple_network::get(std::vector<uint8_t>& _vec)
{   
    std::vector<uint8_t> result = std::move(_vec);
    for (hidden_layer h: layers)
    {
        h.calculate_results(&result[0]);
        result = std::move(h.get_results());
    }
    return result;
}

} 




#endif