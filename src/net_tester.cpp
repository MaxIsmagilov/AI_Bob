#include "AI_Bob_net.hh"
#include <iostream>

int main(int argc, char const *argv[])
{
    AI_BOB::training_network<int8_t> pictures(10,20,20,10);

    pictures.stochastify(0,5);
    pictures.export_as("bob");
    return 0;
}
