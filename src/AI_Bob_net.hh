#ifndef AI_BOB_NET
#define AI_BOB_NET

#include <bits/stdc++.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include <filesystem>
#include <sstream>

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
            std::copy(vec.begin(), vec.end()-1,
                std::ostream_iterator<vectype>(oss, ","));
            oss << vec.back();
        } return oss.str() + "\n";}
public:

        template <typename... params>
    explicit simple_network(int size1, int size2, params... args) : __input_size{size1}, __output_size{std::get<sizeof...(params)-1>(std::forward_as_tuple(args...))}
     {  std::vector<int> vec = {size1,size2, ((int) args)...};
        for (int i = 1; i < vec.size(); i++)
            layers.push_back(hidden_layer(vec[i],vec[i-1]));}

    std::vector<T> get(std::vector<T>& _vec);

    void import_from(std::string name)  
     {  std::ifstream in(name);}

    void export_as(std::string name)
     {  const fs::path pth{"./" + name};
        if (!fs::exists(pth))
            fs:create_directory(pth);
        std::string buf = std::to_string(__input_size) + "\n";   
        std::for_each(layers.begin(), layers.end(), [&](const hidden_layer& h) {buf += std::to_string(h.get_size()) + '\n';});
        buf += "\n";
        std::for_each(layers.begin(), layers.end(), [&](const hidden_layer& h) 
         {  const std::string wstr(vec_to_string(h.get_weights()));
            const std::string bstr(vec_to_string(h.get_biases() ));
            buf += wstr + bstr; });
        std::ofstream out(pth/"data");
        out << buf; }

protected:

    class hidden_layer;
    int __input_size{0};
    int __output_size{0};
    std::vector<hidden_layer> layers;
    struct wb_pair
    {
        int weight;
        int bias;

        /*static void operator delete(void* ptr)
         {  ::operator delete(ptr); }

        static void operator delete[](void* ptr, std::size_t sz)
         {  ::operator delete(ptr); }*/
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
     { __results = std::move(values);}

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

} 




#endif