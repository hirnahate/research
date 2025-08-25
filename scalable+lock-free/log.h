//written by professor cuneo
#ifndef TEST_LOG
#define TEST_LOG

#include <iostream>
#include <sstream>

class Log : public std::stringstream {
    bool enabled;
    public:
    Log();
    Log(bool enabled);
    void flush();
    ~Log();
};

#endif // TEST_LOG