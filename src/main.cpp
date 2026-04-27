#include "engine.h"
#include "utils/logger.h"

int main(int argc, char** argv)
{
    fox_tracer::log::init();
    fox_tracer::engine eng;
    return eng.run(argc, argv);
}
