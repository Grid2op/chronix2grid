#include <algorithm>

char* getCmdOption(char ** begin, char ** end, const string& option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const string& option)
{
    return std::find(begin, end, option) != end;
}