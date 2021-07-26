/**
 * \file utilities_sc.h
*/
#ifndef _UTILITIES_SC_H_
#define _UTILITIES_SC_H_
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
class Timer{
    private:
        std::chrono::steady_clock::time_point last;
    public:
        Timer():last{std::chrono::steady_clock::now()} { }
        double printDiff(const std::string& msg = "Timer diff: ") {
            auto now{std::chrono::steady_clock::now()};
            std::chrono::duration<double, std::milli> diff{now - last};
            std::cout << msg <<std::fixed << std::setprecision(2) << diff.count() << " [ms]\n";
            last = std::chrono::steady_clock::now();
            return diff.count();
        }
};
inline std::vector<std::string> arguments(int argc, char* argv[]){
    std::vector<std::string> res; 
    for (int i = 0; i!=argc; ++i) res.push_back(argv[i]);  
    return res;
}
// 输出进度条
inline void progress_bar(int n)
{
    if (n == 0) return;
    else if (n == 1)
    { 
        std::cout <<"\n ";
    } 
    else if(n<11) 
    {
        std::cout <<"\b\b\b";
    }
    else
    {
        std::cout << "\b\b\b\b";
    }
    std::cout <<"> "<<n<<"%";      
	std::cout.flush();
} 
#endif 
