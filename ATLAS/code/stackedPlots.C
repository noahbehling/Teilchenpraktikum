#include "fileHelper.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TStyle.h"
#include <iostream>
#include "string.h"
#include "TH1F.h"
#include "THStack.h"
#include "TLatex.h"
#include "TLegend.h"

using namespace std;

void SetStyle();
void PlotStack(TString path, TString varUnit, THStack * mc, TH1F * data, TLegend * legend);

int main() {
  ////////////////////////////////////////////////////////////////////////////////
  // This file can guide you through the process of combining monte carlo and data
  ///////////////////////////////////////////////////////////////////////////////

  // Let's say you want to have all the relevant plots in one file, this is how you create it
  TFile * analysis = new TFile("analysis.root", "RECREATE");

  // wähle die betrachtete Diskriminante
  //TString dis = "m_event";
  //TString name = "dis4";

  TString dis[19] = {"lep_pt", "lep_eta", "lep_phi", "lep_E", "jet_pt", "jet_eta", "jet_phi", "jet_E", "jet_goot", "jet_b_tagged", "met_et", "jet_pt_max", "jet_eta_max", "jet_phi_max", "jet_E_max","del_phi","m_jets_pt","m_event", "Eta_event"};
  TString name[19] = {"lep_pt", "lep_eta", "lep_phi", "lep_E", "jet_pt", "jet_eta", "jet_phi", "jet_E", "jet_good", "jet_btag", "met_et", "jet_pt_max", "jet_eta_max", "jet_phi_max", "jet_E_max", "del_phi", "dis3", "dis4", "dis5"};
  string label[19] = {"p_{T}(l) [MeV]", "Eta(l)", "Phi(l)", "E(l) [MeV]", "p_{T}(jets) [MeV]", "Eta(jets)", "Phi(jets)", "E(jets) [MeV]", "#good jets", "#b-tagged good jets", "missing E_{T} [MeV]", "p_{T, max}(jets) [MeV]", "Eta (max p_T jet)", "Phi (max p_T jet)", "E (max p_T jet) [MeV]", "Delta Phi (l)", "m of 3 max p_T jets [MeV]", "m of 4 max p_T jets, lepton, neutrino [MeV]", "Eta of 4 max p_T jets, lepton, neutrino"};

  // now you have to get the histograms you want to use. If you have saved them in another file beforehand, you can get them this way:

  for (int i = 0; i < 19; i++){
  // lade die Daten für Vergleich mit MC
  TFile * file_histogram = new TFile("plots/data_root/data.el_" + name[i] + "_hist.root");
  TH1F * h_histogram = (TH1F*)file_histogram->Get(dis[i]);

  // lade MC Hintergründe
  // ttbar
  TFile * file_histogramttbar = new TFile("plots/MC_root/ttbar.el_" + name[i] + "_hist.root");
  TH1F * h_histogramttbar = (TH1F*)file_histogramttbar->Get(dis[i]);
  h_histogramttbar->Scale(3.22148068e-02); // Fehler

  // singletop
  TFile * file_histogramst = new TFile("plots/MC_root/singletop.el_" + name[i] + "_hist.root");
  TH1F * h_histogramst = (TH1F*)file_histogramst->Get(dis[i]);
  h_histogramst->Scale(3.57195859e-02);

  // diboson
  TFile * file_histogramdib = new TFile("plots/MC_root/diboson.el_" + name[i] + "_hist.root");
  TH1F * h_histogramdib = (TH1F*)file_histogramdib->Get(dis[i]);
  h_histogramdib->Scale(3.18800331e-02); // fehler

  // zjets
  TFile * file_histogramzj = new TFile("plots/MC_root/zjets.el_" + name[i] + "_hist.root");
  TH1F * h_histogramzj = (TH1F*)file_histogramzj->Get(dis[i]);
  h_histogramzj->Scale(6.72368590e-02);

  // wjets
  TFile * file_histogramwj = new TFile("plots/MC_root/wjets.el_" + name[i] + "_hist.root");
  TH1F * h_histogramwj = (TH1F*)file_histogramwj->Get(dis[i]);
  h_histogramwj->Scale(5.44274966e-01);


  //If you want to scale the histogram, use Scale(float factor)
  //If you want to adjust the bin width, use Rebin(int number_of_bins_to_be_merged)

  //You should set a different fill color for each process using SetFillColor(Color_t fcolor); examples for fcolor are kRed, kGreen, kYellow etc. 
  //  You can also add integer like kRed+2 to change the shade

  h_histogram->SetLineColor(kBlack);

  h_histogramttbar->SetFillColor(kRed);
  h_histogramst->SetFillColor(kGreen);
  h_histogramdib->SetFillColor(kYellow);
  h_histogramzj->SetFillColor(kBlue);
  h_histogramwj->SetFillColor(kCyan);

  //You might also want to change the line color using e.g. SetLineColor(kBlack)
  
  //You should add a legend to be able to distinguish the different processes
  TLegend *leg = new TLegend(0.7,0.6,0.85,0.9);
  //leg->SetFillColor(0);
  //leg->SetBorderSize(0);
  //leg->SetTextSize(0.035);
  leg->AddEntry(h_histogram,"data", "f");

  leg->AddEntry(h_histogramttbar,"ttbar", "f");
  leg->AddEntry(h_histogramst,"single top", "f");
  leg->AddEntry(h_histogramdib,"diboson", "f");
  leg->AddEntry(h_histogramzj,"Z+Jets", "f");
  leg->AddEntry(h_histogramwj,"W+Jets", "f");

  //To plot several MC histograms, use THStack. At this point you should be able to figure out its use by looking it up online. 
  //For further analysis, you should however combine them to a new TH1F

  THStack *hs = new THStack("hs","");
  hs->Add(h_histogramttbar);
  hs->Add(h_histogramst);
  hs->Add(h_histogramdib);
  hs->Add(h_histogramzj);
  hs->Add(h_histogramwj);

//  TH1F *h_MC =(TH1F*)hs->GetHistogram();
    
  //For histograms of data, you can use the following commands to change the attributes of the histobramm
  //h_data->SetLineWidth(0);
  //h_data->SetLineColor(kBlack);
  //h_data->SetMarkerStyle(7);
  //h_data->SetMarkerSize(1);
  //h_data->SetMarkerColor(kBlack);

  //For plotting data and stacked MC, you can use the function PlotStack at the end of this file 

  PlotStack("plots/stacked/"+ dis[i] + ".pdf", label[i], hs, h_histogram, leg);

  if (i == 17){
  analysis->cd();
  //For all objects you want to write to the analysis file, call Write(), e.gl
  h_histogram->Write();

  h_histogramttbar->Write();
  h_histogramst->Write();
  h_histogramdib->Write();
  h_histogramzj->Write();
  h_histogramwj->Write();

  hs->Write();

  //TFile * file_histMC = new TFile("plots/MC_root/ttbar.el_" + name[i] + "_hist.root");
  TH1F * h_MC = (TH1F*)h_histogramttbar->Clone();
  //h_MC->Scale(3.22148068e-02); 
  //h_MC->Add(h_histogramttbar);
  h_MC->Add(h_histogramst);
  h_MC->Add(h_histogramdib);
  h_MC->Add(h_histogramzj);
  h_MC->Add(h_histogramwj);

  // save all backgrounds
  fileHelper::SaveNewHist("plots/MC_backgrounds.root", h_MC, true);
  }

  //End the programm properly by deleting all dynamic instances
  file_histogram->Close();
  delete file_histogram;
  delete leg;
  }

  //delete h_histogram;
  //delete h_histogramttbar;
  //delete h_histogramst;
  //delete h_histogramdib;
  //delete h_histogramzj;
  //delete h_histogramwj;
  analysis->Close();



  return 0;
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
  gStyle->SetHistLineWidth(1);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);
}


void PlotStack(TString filename, TString varUnit, THStack* mc, TH1F * data, TLegend * legend) {
  SetStyle();
	TCanvas * canv = new TCanvas("canv","Canvas for histogram",1);
  canv->SetLeftMargin(.12);
  canv->SetRightMargin(.1);
  data->Draw("E1");
  data->SetMinimum(0);
  mc->Draw("hist SAME");
  data->Draw("E1 SAME");
  mc->SetTitle(";"+varUnit+";Events");
  mc->GetYaxis()->SetTitleOffset(1);
  data->Draw("E1 SAME");
  legend->Draw();
  canv->Print(filename);
  delete canv;
}






