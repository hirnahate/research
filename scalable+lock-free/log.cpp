//written by professor cuneo
#include "log.h"

Log::Log(bool enabled)
    : enabled(enabled)
{
    if (!enabled) {
        setstate(std::ios::failbit);
    }
}

Log::Log() : Log(true) {}

void Log::flush() {
    std::cout << str();
    str("");
    clear();
}

Log::~Log() {
    if (enabled) {
        flush();
    }
}
