#include "fileHelper.h"
#include "TSystem.h"
#include <iostream>
#include "TFile.h"

using namespace std;

mini * fileHelper::GetMiniTree(string path) {
  TChain * chain = new TChain("mini");
  Long_t tmp_a, tmp_b, tmp_c, tmp_d;
  if (gSystem->GetPathInfo(path.data(), &tmp_a, &tmp_b, &tmp_c, &tmp_d) != 0) {
    cout << "ERROR no file present in path " << path << endl;
    return 0;
  }
  chain->AddFile(path.data());
  return new mini(chain);
}

void fileHelper::SaveNewHist(std::string path, TH1F * hist, bool recreate) {
  TString str = path;
  str.ReplaceAll(".root", "_hist.root");
  TFile * newFile;
  if(recreate){
    newFile = new TFile(str, "RECREATE");
  }
  else{
    newFile = new TFile(str, "UPDATE");
  }
  hist->Write();
  newFile->Close();
  delete newFile;
}
