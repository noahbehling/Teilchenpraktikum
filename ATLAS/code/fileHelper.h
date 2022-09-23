#ifndef fileHelper_h
#define fileHelper_h

#include "mini.h"
#include "string.h"
#include "TTree.h"
#include "TH1F.h"

class fileHelper {

 public:
  static mini * GetMiniTree(std::string path);

  static void SaveNewHist(std::string path, TH1F * hist, bool recreate);
  // Saves histogram hist in file with name path. For Example path = "example.root". 
  // If recreate is true, a new file is created, if it is false, the histogram is appended.
  
};

#endif
