#include "mini.h"
#include "fileHelper.h"
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main(int argn, char *argv[]) {

  //if you want to run this script for several input files, it is useful to call the name of the file when executing this program
  if(argn==1){
    cout << "Please start runSelection.exe with added argument of file to be processed" << endl;
    return 1;
  }
  // path to the file to be studied + filename from argument called when executing file
  string inpath = string(argv[1]);
  TString filename = TString(inpath).ReplaceAll("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/zprime/E4/Final/samples/", "");

  cout << "Processing " << filename << endl;

  // retrieve the tree from the file
  mini * tree = fileHelper::GetMiniTree(inpath);
  if (tree == 0) {
    cout << "ERROR tree is null pointer" << endl;
    return 1;
  }

  // check that the tree is not empty
  int nEntries = tree->GetEntries();
  cout << "INFO tree contains " << nEntries << " events" << endl;
  if (nEntries == 0) {
    cout << "ERROR tree contains zero events" << endl;
    return 1;
  }

  // create file to which the selection is going to be saved to
  TString outpath = "output_runSelection/" + filename;
  outpath.ReplaceAll(".root", "_selected.root");
  TFile * newFile = new TFile(outpath, "RECREATE");
  
  // make a copy of the tree to be saved after selection
  bool passCriteria;
  TTree * newTree = tree->CloneTree();

  // define variables to count the efficiency
  int cut_lep_n = nEntries;
  int cut_jet_n = nEntries;
	int cut_btag = nEntries;
	int cut_met = nEntries;
	int cut_lep_pt = nEntries;
	int cut_jet_pt = nEntries;
  int cut_btag_pt = nEntries;
  int cut_jet_good_pt = nEntries;

  // initialize variables to save values for event

	int b_tagged_n;
  bool b_tagged_pt;
	int jet_good;
	bool jet_good_pt;
  bool lep_pt_n;
   

  // now loop over the events in the tree
  for (int iEntry = 0; iEntry < nEntries; ++iEntry) {
   
    // get entry no. iEntry and tell every 100000th event
    tree->GetEntry(iEntry);
    if ((iEntry+1)%100000 == 0) {
      cout << "INFO processing the " << iEntry+1 << "th event" << endl;
    }

    //////////////////////////////////////////////////////
    // To do: Implement all required selection criteria //
    //////////////////////////////////////////////////////
    passCriteria = true;    


	// count the b-tagged jets
	b_tagged_n = 0;
	jet_good = 0;
	jet_good_pt = false;
  b_tagged_pt = false;
  lep_pt_n = false;

	for(UInt_t i = 0; i<tree->jet_n; i++){
		if (tree->jet_MV1[i] > 0.7892 && tree->jet_good[i] == true){ 
      b_tagged_n++;
      if (tree->jet_pt[i] > 50000){ b_tagged_pt = true; }
    }
		if (tree->jet_good[i] == true) {  jet_good++; }
		if (tree->jet_pt[i] > 80000 && tree->jet_good[i] == true){ jet_good_pt = true; } // 50000
	}

  for(UInt_t i = 0; i<tree->lep_n; i++){

      if (tree->lep_pt[i] > 50000){ lep_pt_n = true; }
	}




    if(jet_good < 4){ 
	passCriteria = false;
	//cut_lep_n--;
	cut_jet_n--;
	//cut_btag--; 
	//cut_met--;
	//cut_lep_pt--;
  //cut_btag_pt--;
  //cut_jet_good_pt--;
    }
  if(tree->lep_n != 1){ 
	passCriteria = false;
	cut_lep_n--;
	//cut_btag--;
	//cut_met--;
	//cut_lep_pt--;
  //cut_btag_pt--;
  //cut_jet_good_pt--;
    }  
  if (tree->met_et < 40000) {
	passCriteria = false;
	cut_met--;
	//cut_btag--; 
	//cut_lep_pt--;
  //cut_btag_pt--;
  //cut_jet_good_pt--;
    } 
  if (lep_pt_n == false) { // 40000
	passCriteria = false;
	cut_lep_pt--;
	//cut_btag--;
  //cut_btag_pt--;
  //cut_jet_good_pt--;
    } 
  if (jet_good_pt == false){
	passCriteria = false;
	//cut_btag--;
	cut_jet_good_pt--;
  //cut_btag_pt--;
    } 
  if (b_tagged_n<2) {
	passCriteria = false;
	cut_btag--;
  //cut_btag_pt--;
    } 
  if (b_tagged_pt == false){
  passCriteria = false;
  cut_btag_pt--;
    }  

    // check all selection criteria and only save the event to the new
    // tree if all of them are true
    if (passCriteria) {
      newTree->Fill();
    }
    
  }






  // write all efficiencies for the file into efficiencies.txt
  
  ofstream eff ("output_runSelection/efficiencies.txt", ios_base::app);
  if (eff.is_open()) {
	eff << filename << " ";

  eff << cut_jet_n / float(nEntries)  << " ";
  eff << cut_lep_n / float(nEntries)  << " ";
  eff << cut_met / float(nEntries)  << " ";
  eff << cut_lep_pt / float(nEntries)  << " ";
  eff << cut_jet_good_pt / float(nEntries)  << " ";
  eff << cut_btag / float(nEntries)  << " ";
  eff << cut_btag_pt / float(nEntries)  << " ";



  eff << " " << endl;
	eff.close();
  } else {
  cout << "unable to open efficiencies.txt";
  }


  // save new tree to file
  cout << "INFO saving new tree with " << newTree->GetEntries() << " entries" << endl;
  newFile->Write();
  gDirectory->Purge();
  newFile->Close();
  
  // end program happily
  delete newFile;
  return 0;
}
