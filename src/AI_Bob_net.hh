#ifndef AI_BOB_NET
#define AI_BOB_NET

#include <bits/stdc++.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include <filesystem>
#include <sstream>
#include <random>

namespace AI_BOB
{

namespace fs = std::filesystem;

    template <typename T>
class simple_network
{
private:
        template <typename vectype>
    std::string vec_to_string(const std::vector<vectype>& vec)
        {  std::ostringstream oss;
        if (!vec.empty())
        {
            std::copy(vec.begin(), vec.end(),
                std::ostream_iterator<vectype>(oss, ","));
        } return oss.str() + "\n";}
public:

        template <typename... params>
    explicit simple_network(int size1, int size2, params... args) : __input_size{size1}, __output_size{std::get<sizeof...(params)-1>(std::forward_as_tuple(args...))}
     {  std::vector<int> vec = {size1,size2, ((int) args)...};
        for (int i = 1; i < vec.size(); i++)
            layers.push_back(hidden_layer(vec[i],vec[i-1]));}

    std::vector<T> get(std::vector<T>& _vec);

    void import_from(std::string name)  
    {  
        const fs::path pth{"./" + name + "/data"};
        std::vector<int> sizes{};
        std::ifstream in;
        in.open(pth);
        std::string str;
        if (in.is_open()) ;
        while (getline(in, str))
        {
            if (str[0] == (char)(0)) break;
            sizes.push_back(std::stoi(str));
        }
        __input_size = sizes.front();
        __output_size = sizes.back();
        layers.clear();
        for (int i = 1; i < sizes.size(); i++)
            layers.push_back(hidden_layer(sizes[i], sizes[i-1]));
        

        for (int i = 0; i < layers.size(); i++)
        {
            std::vector<int> weights;
            std::vector<int> biases;
            getline(in, str);
            const std::string delimiter = ",";
            size_t pos = 0;
            std::string token;
            while ((pos = str.find(delimiter)) != std::string::npos) 
            {
                token = str.substr(0, pos);
                weights.push_back(std::stoi(token));
                str.erase(0, pos + delimiter.length());
            }
            getline(in, str);
            pos = 0;
            while ((pos = str.find(delimiter)) != std::string::npos) 
            {
                token = str.substr(0, pos);
                biases.push_back(std::stoi(token));
                str.erase(0, pos + delimiter.length());
            }
            std::vector<wb_pair> wbs;
            for (int i = 0; i < weights.size() && i < biases.size(); i++)
            {
                wbs.push_back({weights[i], biases[i]});
            }
            layers.at(i).import_values(wbs);
        }

        in.close();
    }

    void export_as(std::string name)
     {  const fs::path pth{"./" + name};
        if (!fs::exists(pth))
            fs:create_directory(pth);
        std::string buf = std::to_string(__input_size) + "\n";   
        std::for_each(layers.begin(), layers.end(), [&](const hidden_layer& h) {buf += std::to_string(h.get_size()) + '\n';});
        buf += "\n";
        std::for_each(layers.begin(), layers.end(), [&](const hidden_layer& h) 
            {  
                const std::string wstr(vec_to_string(h.get_weights()));
                const std::string bstr(vec_to_string(h.get_biases() ));
                buf += wstr + bstr; 
            });
        std::ofstream out(pth/"data");
        out << buf; 
        out.close();}

protected:

    class hidden_layer;
    int __input_size{0};
    int __output_size{0};
    std::vector<hidden_layer> layers;
    struct wb_pair
    {
        int weight;
        int bias;
    };


};

template<typename T> class simple_network<T>::hidden_layer
{
public:
    explicit hidden_layer(const int this_size, const int prev_size): __tsize{this_size}, __psize{prev_size} 
     {  __results = std::vector<T>();
        __unwrapped_table = std::vector<wb_pair>(); 
        __results.resize(sizeof(T) * this_size);
        __unwrapped_table.resize(sizeof(T) * (this_size * prev_size));   }
    
    explicit hidden_layer(const hidden_layer& other) : __tsize{other.__tsize}, __psize{other.__psize} 
     {  __results = std::move(other.__results); 
        __unwrapped_table = std::move(other.__unwrapped_table); }

    void import_values(const std::vector<wb_pair>& values) 
     { __unwrapped_table = std::move(values);}

    void calculate_results(T* start)
    {  
        for (int i = 0; i < __tsize; i++)
        {
            int temp = 0;
            for (int j = 0; j < __psize; j++)
            {
                temp += trim(apply_wb(i, j, start[j]));
            }
            __results[i] = trim(temp);
        }
    }

    void stochastify(const int amount)
    {
        std::random_device r;
        std::mt19937 rando(r());
        std::uniform_int_distribution<int> uniform_dist(-amount, amount);
        std::for_each(__unwrapped_table.begin(), __unwrapped_table.end(), [&](wb_pair& mb) mutable
            {
                int num = uniform_dist(rando);
                mb.weight += (abs((mb.weight + num) - mb.weight) == abs(num)) ? num : 0;
                num = uniform_dist(rando);
                mb.bias += (abs((mb.weight + num) - mb.weight) == abs(num)) ? num : 0;
            });
    }

    void init_random()
    {
        std::random_device r;
        std::mt19937 rando(r());
        std::uniform_int_distribution<int> uniform_dist(-(0xFFFFFF), 0xFFFFFF);
        std::for_each(__unwrapped_table.begin(), __unwrapped_table.end(), [&](wb_pair& mb) mutable
            {
                int num = uniform_dist(rando);
                mb.weight = num;
                num = uniform_dist(rando);
                mb.bias num;
            });
    }

    std::vector<T> get_results()
     {  return std::move(__results); }

    int get_size()
     {  return __tsize;  }

    int get_size() const
     {  return __tsize;  }
 
    std::vector<int> get_weights() const
     {  std::vector<int> vec{}; 
        std::for_each(__unwrapped_table.begin(), __unwrapped_table.end(), [&](const wb_pair& i) {vec.push_back(i.weight);}); 
        return std::move(vec);  }

    std::vector<int> get_biases() const
     {  std::vector<int> vec{}; 
        std::for_each(__unwrapped_table.begin(), __unwrapped_table.end(), [&](const wb_pair& i) {vec.push_back(i.bias);}); 
        return std::move(vec);  }

private:

    inline int apply_wb(int this_index, int other_index, T value)
     {  const wb_pair& wb = __unwrapped_table[this_index + (other_index + this_index)];
        return ((int) value) * wb.weight + wb.bias;        }

    inline T trim(int value)
     {  const int s = (int)sizeof(T)-1;
        const uint64_t exponent = 1ULL << std::max((((uint64_t)value + 0x16000000ULL) / 0x1908B100ULL),0ULL); 
        return (T) (((8ULL * (1ULL << s) * exponent)/(8+exponent)) / 8ULL); }

    std::vector<wb_pair> __unwrapped_table;
    std::vector<T> __results;
    const int __tsize{0};
    const int __psize{0};
};

template<typename T> std::vector<T> simple_network<T>::get(std::vector<T>& _vec)
{   
    std::vector<T> result = std::move(_vec);
    for (hidden_layer h: layers)
    {
        h.calculate_results(&result[0]);
        result = std::move(h.get_results());
    }
    return result;
}

template<typename T> class training_network : public simple_network<T>
{
public: 

    struct training_data_entry
    {
        std::vector<T> input;
        std::vector<T> output;
    };

        template <typename... params>
    explicit training_network(int size1, int size2, params... args) : simple_network<T>(size1, size2, args...) {}
    
    void stochastify(int layer, int amount)
    {
        simple_network<T>::layers.at(layer).stochastify(amount);
    }

    void add_training_data(training_data_entry& t)
    {
        datas.push_back(std::move(t));
    }

    void randomize()
    {
        
    }

    void train()
    {

    }

protected:

    static constexpr int num_of_checked_layers = 100;

    std::vector<T> replace_layer_get(std::vector<T>& _vec, const class simple_network<T>::hidden_layer& replace_layer, int num)
    {
        std::vector<class simple_network<T>::hidden_layer> newlayers = std::move(_vec);
        newlayers[num] = std::move(replace_layer);
        std::vector<T> result = std::move(_vec);
        for (class simple_network<T>::hidden_layer h: newlayers)
        {
            h.calculate_results(&result[0]);
            result = std::move(h.get_results());
        }
        return result;
    }

    int get_cost(const training_data_entry& expected_result, const std::vector<T>& actual_result)
    {
        std::vector<int> cost_list;
        for (int i = 0; i < actual_result.size(); i++)
        {
            cost_list.push_back((int) pow(expected_result.output[i] - actual_result[i], 2));
        }
        int sum;
        std::accumulate(cost_list.begin(), cost_list.end(), &sum);
        return sum;
    }

    void pass_over_layer(int layer, int scale_factor)
    {
        struct cost_pair 
        {
            class simple_network<T>::hidden_layer hl;
            int cost;
            bool operator<(const cost_pair& cp)
            {
                return this->cost < cp.cost;
            }
        }; 
        std::vector<cost_pair> test_layers;
        for (int i = 0; i < num_of_checked_layers; i++)
        {
            test_layers.push_back({std::move(simple_network<T>::layers[layer]),0});
            test_layers.at(i).hl.stochastify(scale_factor);
        }
    }

    std::vector<training_data_entry> datas;

};
 

} 




#endif