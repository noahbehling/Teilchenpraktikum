#include "fileHelper.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TStyle.h"
#include <iostream>
#include "string.h"
#include "TH1F.h"
#include "TLatex.h"
#include "TLorentzVector.h"
#include "TLegend.h"

#include <cmath>
#include <math.h>
#include <string>

using namespace std;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// These functions might be useful. They can be found at the end of this script and don't have to be but can be altered. //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SetStyle();
// Should be called before plotting to get a decent looking plot

TH1F * InitHist(TString varName,TString varUnit, int numberBins, float minBin, float maxBin, bool isData); 
// Helps you initialize a histogram so that it already ha the correct labels when plotted.
// varName is the variable name in the tuple or the name you pick for a derived physical variable. You should stick to it for the rest of the analysis. (e.g. lep_eta)
// 

void PlotHist(TString filename, TH1F * hist);
// Creates a file in with path name (e.g. "text.pdf") and plots histogram in it

 


int main(int argn, char *argv[]) {

  //if you want to run this script for several input files, it is useful to call the name of the file when executing this program
  if(argn==1){
    cout << "Please start plotDistribution.exe with added argument of file to be processed" << endl;
    return 1;
  }

  // path to the file to be studied, e.g.
  string path = string(argv[1]);
  // is the file a data file or not? setting this variable now might be useful
  bool isdata = false;

  // retrieve the tree from the file
  mini * tree = fileHelper::GetMiniTree(path);
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
  
  //////////////////////////////////////////////////////////////////////////////
  // To do: initialize histograms to be made
  // example:
  TH1F * h_lep_pt = InitHist("lep_pt","p_{T}(l) [MeV]",25,50.e3,200.e3,isdata);
  TH1F * h_lep_eta = InitHist("lep_eta", "eta", 25, -2.5, 2.5, isdata);
  TH1F * h_lep_phi = InitHist("lep_phi", "phi", 25, -3.12, 3.12, isdata);
  TH1F * h_lep_E = InitHist("lep_E", "E(l) [MeV]", 25, 30.e3, 300.e3, isdata);

  TH1F * h_jet_pt = InitHist("jet_pt","p_{T}(l) [MeV]",50,25.e3,200.e3,isdata);
  TH1F * h_jet_eta = InitHist("jet_eta", "eta", 50, -2.5, 2.5, isdata);
  TH1F * h_jet_phi = InitHist("jet_phi", "phi", 25, -3.12, 3.12, isdata);
  TH1F * h_jet_E = InitHist("jet_E", "E(l) [MeV]", 50, 25.e3, 200.e3, isdata);

  TH1F * h_jet_good = InitHist("jet_goot", "number of good jets in event", 9, 4, 12, isdata);
  TH1F * h_jet_btag = InitHist("jet_b_tagged", "number of b-tagged jets in event", 4, 2, 6, isdata);

  TH1F * h_met_et = InitHist("met_et", "p_{T, miss.} [MeV]", 25, 40.e3, 200.e3, isdata);

  TH1F * h_jet_pt_max = InitHist("jet_pt_max","p_{T}(l) [MeV]",25,80.e3,200.e3,isdata);
  TH1F * h_jet_eta_max = InitHist("jet_eta_max", "eta", 25, -2.5, 2.5, isdata);
  TH1F * h_jet_phi_max = InitHist("jet_phi_max", "phi", 25, -3.12, 3.12, isdata);
  TH1F * h_jet_E_max = InitHist("jet_E_max", "E(l) [MeV]", 25, 40.e3, 300.e3, isdata);

  TH1F * h_del_phi = InitHist("del_phi", "Delta phi", 25, 0, 3.12, isdata);
  TH1F * h_dis3 = InitHist("m_jets_pt", "m of three jets with largest p_{T} [MeV]", 25, 0, 150.e3, isdata);
  TH1F * h_dis4 = InitHist("m_event", "m of event [MeV]", 25, 150.e3, 3200.e3, isdata);
  TH1F * h_dis5 = InitHist("Eta_event", "Eta of event", 25, -3, 3, isdata);

  //
  //

  //////////////////////////////////////////////////////////////////////////////////

  float lep_pt;
  float lep_eta;
  float lep_phi;
  float lep_E;

  float jet_pt;
  float jet_eta;
  float jet_phi;
  float jet_E;

  int jet_good;
  int jet_btag;

  int pt_max;

  float del_phi;
  float del_phi_1;
  
  int i_max_pt[4] = {0, 0, 0, 0}; // inices of four jets with largest pt 

  TLorentzVector v1;
  TLorentzVector v2;
  TLorentzVector v3;
  TLorentzVector v4; // lerentz vector for invariant mass of three largest pt jets

  TLorentzVector max3;
  TLorentzVector max4;

  float inv_mass3;

  TLorentzVector met;
  TLorentzVector lep;
  TLorentzVector nu;

  float inv_mass_all;

  // now loop over the events in the tree
  for (int iEntry = 0; iEntry < nEntries; ++iEntry) {

    // get entry no. iEntry and tell every 100th event
    tree->GetEntry(iEntry);
    if ((iEntry+1)%10000 == 0) {
      cout << "INFO processing the " << iEntry+1 << "th event" << endl;
    }

    // For Monte Carlo, each event has to have a scale factor. 
    // The scale factors necessary can be found separately in the samples, but they have also been combined in the variable scaleFactor_COMBINED.
    float w = 1.;
    if (!isdata)
      w = tree->scaleFactor_COMBINED;
    
    // plot all leptons
    /////////////////////////////////////////////////////////////////////////////////////////
    // Get variable or calculate it (s. instructions)
    lep_pt = tree->lep_pt[0];
    lep_eta = tree->lep_eta[0];
    lep_phi = tree->lep_phi[0];
    lep_E = tree->lep_E[0];
    //
    ///////////////////////////////////////////////////////////////////////////////////////////  
    // fill histograms
    h_lep_pt->Fill(lep_pt,w);
    h_lep_eta->Fill(lep_eta,w);
    h_lep_phi->Fill(lep_phi,w);
    h_lep_E->Fill(lep_E,w);
    //
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    del_phi = abs(tree->lep_phi[0] - tree->met_phi);
    del_phi_1 = abs(2 * M_PI - del_phi);
    if(del_phi > del_phi_1){ del_phi = del_phi_1;}

    h_del_phi->Fill(del_phi,w);
    

    jet_good = 0;
    jet_btag = 0;
    pt_max = 0;


    for (UInt_t i=0; i < tree->jet_n; i++){
      if (tree->jet_good[i] == true){
        jet_good++;
        if (tree->jet_MV1[i] > 0.7892){ jet_btag++; }
	if (tree->jet_pt[i] > tree->jet_pt[pt_max]){ pt_max = i; }

        jet_pt = tree->jet_pt[i];
        jet_eta = tree->jet_eta[i];
        jet_phi = tree->jet_phi[i];
        jet_E = tree->jet_E[i];

        h_jet_pt->Fill(jet_pt,w);
        h_jet_eta->Fill(jet_eta,w);
        h_jet_phi->Fill(jet_phi,w);
        h_jet_E->Fill(jet_E,w);
      }
    }
   
    // save pt of all jets
    float arr_jet_pt[tree->jet_n];
    for (UInt_t i=0; i < tree->jet_n; i++){
        arr_jet_pt[i] = tree->jet_pt[i];
    }

    // save largest good jet
    i_max_pt[0] = pt_max;
    arr_jet_pt[pt_max] = 0;

    // save second largest good jets 
    for (UInt_t i=0; i < tree->jet_n; i++){
      if (tree->jet_good[i] == true){
        if (tree->jet_pt[i] > tree->jet_pt[i_max_pt[1]]){ i_max_pt[1] = i; }
        }
    }
    arr_jet_pt[i_max_pt[1]] = 0;

    // save third largest good jet 
    for (UInt_t i=0; i < tree->jet_n; i++){
      if (tree->jet_good[i] == true){
        if (tree->jet_pt[i] > tree->jet_pt[i_max_pt[2]]){ i_max_pt[2] = i; }
        }
    }
    arr_jet_pt[i_max_pt[2]] = 0;

    // save fourth largest good jet 
    for (UInt_t i=0; i < tree->jet_n; i++){
      if (tree->jet_good[i] == true){
        if (tree->jet_pt[i] > tree->jet_pt[i_max_pt[3]]){ i_max_pt[3] = i; }
        }
    }

    // save sum of three largest pt lorentz vectors
    v1.SetPtEtaPhiE(tree->jet_pt[i_max_pt[0]], tree->jet_eta[i_max_pt[0]],tree->jet_phi[i_max_pt[0]],tree->jet_E[i_max_pt[0]]);
    v2.SetPtEtaPhiE(tree->jet_pt[i_max_pt[1]], tree->jet_eta[i_max_pt[1]],tree->jet_phi[i_max_pt[1]],tree->jet_E[i_max_pt[1]]);
    v3.SetPtEtaPhiE(tree->jet_pt[i_max_pt[2]], tree->jet_eta[i_max_pt[2]],tree->jet_phi[i_max_pt[2]],tree->jet_E[i_max_pt[2]]);
    v4.SetPtEtaPhiE(tree->jet_pt[i_max_pt[3]], tree->jet_eta[i_max_pt[3]],tree->jet_phi[i_max_pt[3]],tree->jet_E[i_max_pt[3]]);

    max3 = v1 + v2 + v3;
    inv_mass3 = max3.M();

    h_dis3->Fill(inv_mass3,w);


    max4 = max3 + v4;

    max4 = max3 + v4;
    lep.SetPtEtaPhiE(tree->lep_pt[0], tree->lep_eta[0],tree->lep_phi[0],tree->lep_E[0]);
    met.SetPhi(tree->met_phi);
    met.SetE(tree->met_et);

    max4 = max4 + lep + met; 
    inv_mass_all = max4.M();

    h_dis4->Fill(inv_mass_all,w);

    h_dis5->Fill(max4.Eta(),w);

    
    h_jet_good->Fill(jet_good,w);
    h_jet_btag->Fill(jet_btag,w);

    h_met_et->Fill(tree->met_et,w);

    h_jet_pt_max->Fill(tree->jet_pt[pt_max],w);
    h_jet_eta_max->Fill(tree->jet_eta[pt_max],w);
    h_jet_phi_max->Fill(tree->jet_phi[pt_max],w);
    h_jet_E_max->Fill(tree->jet_E[pt_max],w);

        
  }

  SetStyle();

  ///////////////////////////////////////////////////////////////////////////////////////////
  //To do: Use PlotHist to plot

  TString filename1 = TString(path).ReplaceAll("output_runSelection/", "");
  filename1 = filename1.ReplaceAll("_selected.root", "");
  string filename = string(filename1);
  PlotHist("plots/pdfs/" + filename + "_lep_pt.pdf", h_lep_pt);
  PlotHist("plots/pdfs/" + filename + "_lep_eta.pdf", h_lep_eta);
  PlotHist("plots/pdfs/" + filename + "_lep_phi.pdf", h_lep_phi);
  PlotHist("plots/pdfs/" + filename + "_lep_E.pdf", h_lep_E);

  PlotHist("plots/pdfs/" + filename + "_jet_pt.pdf", h_jet_pt);
  PlotHist("plots/pdfs/" + filename + "_jet_eta.pdf", h_jet_eta);
  PlotHist("plots/pdfs/" + filename + "_jet_phi.pdf", h_jet_phi);
  PlotHist("plots/pdfs/" + filename + "_jet_E.pdf", h_jet_E);

  PlotHist("plots/pdfs/" + filename + "_jet_good.pdf", h_jet_good);
  PlotHist("plots/pdfs/" + filename + "_jet_btag.pdf", h_jet_btag);

  PlotHist("plots/pdfs/" + filename + "_met_et.pdf", h_met_et);

  PlotHist("plots/pdfs/" + filename + "_jet_pt_max.pdf", h_jet_pt_max);
  PlotHist("plots/pdfs/" + filename + "_jet_eta_max.pdf", h_jet_eta_max);
  PlotHist("plots/pdfs/" + filename + "_jet_phi_max.pdf", h_jet_phi_max);
  PlotHist("plots/pdfs/" + filename + "_jet_E_max.pdf", h_jet_E_max);

  PlotHist("plots/pdfs/" + filename + "_del_phi.pdf", h_del_phi); 
  PlotHist("plots/pdfs/" + filename + "_dis3.pdf", h_dis3); 
  PlotHist("plots/pdfs/" + filename + "_dis4.pdf", h_dis4); 
  PlotHist("plots/pdfs/" + filename + "_dis5.pdf", h_dis5); 


  /////////////////////////////////////////////////////////////////////////////////////////////
  //You can now use fileHelper::SaveNewHist to save histograms

  fileHelper::SaveNewHist("plots/root/" + filename + "_lep_pt.root", h_lep_pt, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_lep_eta.root", h_lep_eta, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_lep_phi.root", h_lep_phi, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_lep_E.root", h_lep_E, true);

  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_pt.root", h_jet_pt, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_eta.root", h_jet_eta, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_phi.root", h_jet_phi, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_E.root", h_jet_E, true);

  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_good.root", h_jet_good, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_btag.root", h_jet_btag, true);

  fileHelper::SaveNewHist("plots/root/" + filename + "_met_et.root", h_met_et, true);

  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_pt_max.root", h_jet_pt_max, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_eta_max.root", h_jet_eta_max, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_phi_max.root", h_jet_phi_max, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_jet_E_max.root", h_jet_E_max, true);

  fileHelper::SaveNewHist("plots/root/" + filename + "_del_phi.root", h_del_phi, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_dis3.root", h_dis3, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_dis4.root", h_dis4, true);
  fileHelper::SaveNewHist("plots/root/" + filename + "_dis5.root", h_dis5, true);

  // To end the program properly, delete all dynamic instances
  delete h_lep_pt;
  delete h_lep_eta;
  delete h_lep_phi;
  delete h_lep_E;

  delete h_jet_pt;
  delete h_jet_eta;
  delete h_jet_phi;
  delete h_jet_E;

  delete h_jet_good;
  delete h_jet_btag;

  delete h_met_et;

  delete h_jet_pt_max;
  delete h_jet_eta_max;
  delete h_jet_phi_max;
  delete h_jet_E_max;

  delete h_del_phi;
  delete h_dis3;
  delete h_dis4;
  delete h_dis5;

  delete tree;

  return 0;
}



//////////////////////////////////////////////////////////////////////////////////////////
////// Functions that can but do not have to be uses: ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////



TH1F * InitHist(TString varName,TString varUnit, int numberBins, float minBin, float maxBin, bool isData){
  TH1F *hist = new TH1F(varName,varName,numberBins,minBin,maxBin);
  hist->SetTitle(";"+varUnit+";Events");
  hist->GetYaxis()->SetTitleOffset(1.3);
  hist->Sumw2(false);
  if(isData){
    hist->Sumw2(true);
  }
  return hist;
}


void PlotHist(TString filename, TH1F * hist){
	TCanvas * canv = new TCanvas("canv","Canvas for histogram",1);
  hist->Draw("hist");
  canv->Print(filename);
  cout << "INFO: " << filename << " has been created" << endl;
  delete canv;
}

void Plot2Hist(TString filename, TString varUnit, TH1F * hist1, TH1F * hist2) {
  TCanvas * canv = new TCanvas("canv","Canvas for histograms",1);
  canv->SetLeftMargin(.12);
  canv->SetRightMargin(.1);
  
  hist1->Draw("HIST");

  hist1->SetTitle(";"+varUnit+";Events");
  hist1->GetYaxis()->SetTitleOffset(1);

  hist2->Draw("HIST SAME");

  TLegend * l = new TLegend(0.5, 0.75, 0.86, 0.9, "");
  l->SetFillColor(0);
  l->SetBorderSize(1);
  l->SetTextSize(0.04);
  l->SetTextAlign(12);
  l->AddEntry(hist1, "Add description", "l");
  l->AddEntry(hist2, "Add description here", "l");
  l->Draw();

  ///////////////////////////////////////////
  // Histograms can be normalized to unit area by calling 
  // hist->Scale(1./hist->Integral()) before plotting
  // In case you decide to do that, you can use the following lines to label your plots
  ////////////////////////////////////////////////////////////////////////////////////////
  //TLatex * t = new TLatex(0.93,0.55,"#bf{Normalized to unit area}");
  //t->SetTextAngle(90);
  //t->SetNDC();
  //t->SetTextSize(.04);
  //t->DrawClone();

  canv->Print(filename);
  cout << "INFO: " << filename << " has been created" << endl;
  delete canv;
}

void SetStyle() {
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetStatColor(0);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetTitleXOffset(1.3);
  gStyle->SetTitleYOffset(1.3);
  gStyle->SetTextSize(0.05);
  gStyle->SetLabelSize(0.05,"x");
  gStyle->SetTitleSize(0.05,"x");
  gStyle->SetLabelSize(0.05,"y");
  gStyle->SetTitleSize(0.05,"y");
  gStyle->SetLabelSize(0.05,"z");
  gStyle->SetTitleSize(0.05,"z");
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
  gStyle->SetHistLineWidth(2);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);
}


