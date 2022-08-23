#ifndef __TIMER_H_
#define __TIMER_H_
#include <sys/time.h>
#include <string>
#include <iostream>

using namespace std;

class Timer {
public:
  Timer(bool _auto_print = false) : auto_print(_auto_print) {
    gettimeofday(&start, NULL);
  }

  ~Timer() {
    if (auto_print) {
      gettimeofday(&end, NULL);
      float interval = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
      std::cout << "Time: " << internal << " ms" << std::endl;
    }
  }

  float getTime() {
    gettimeofday(&end, NULL);
    float interval = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
    return interval;
  }

private:
  bool auto_print;
  struct timeval start;
  struct timeval end;
};

#endif
