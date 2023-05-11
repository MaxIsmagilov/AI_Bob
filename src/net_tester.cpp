#include "AI_Bob_net.hh"
#include <iostream>

int main(int argc, char const *argv[])
{
    AI_BOB::simple_network<int8_t> s(5,5,3);
    s.export_as("bob");
    return 0;
}
