#include <iostream>
#include <string>
#include "TMath.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TLegend.h"


using namespace std;

float chisquareNBins(TH1F * data, TH1F * mc);

void SetStyle();

int main(){

	// At this point, you want to first test the agreement between data and monte carlo.
	// After that, you want to create a limit plot
	// You should be able to do most of it on your own by now, but here are some usefull things:

	// The function chisquareNBins(TH1F * data, TH1F * mc) returns the chi square value that characterizes the aggreement between a data histogram and a monte carlo histogramm
	// It prints the degrees of freedom in this comparison, which might be useful information

	//////////////////////////////////////////////////////////////////
	// Do do: 1. quantify data-mc agreement
	//		  2. calculate limits to the cross section of Z' production
	//
	////////////////////////////////////////////////////////////////////////////

	// deg = 25

	TFile * file_data = new TFile("plots/data_root/data.el_dis4_hist.root");
	TH1F * h_data = (TH1F*)file_data->Get("m_event");

	TFile * file_MC = new TFile("plots/MC_backgrounds_hist.root");
	TH1F * h_MC = (TH1F*)file_MC->Get("m_event");

	float Chi_data_MC = chisquareNBins(h_data, h_MC); 

	cout << Chi_data_MC << endl; 


	TFile * file_z400 = new TFile("plots/MC_root/zprime400.el_dis4_hist.root");
	TH1F * h_z400 = (TH1F*)file_z400->Get("m_event");
	h_z400->Scale(1.1);

	TFile * file_z500 = new TFile("plots/MC_root/zprime500.el_dis4_hist.root");
	TH1F * h_z500 = (TH1F*)file_z500->Get("m_event");
	h_z500->Scale(0.82);

	TFile * file_z750 = new TFile("plots/MC_root/zprime750.el_dis4_hist.root");
	TH1F * h_z750 = (TH1F*)file_z750->Get("m_event");
	h_z750->Scale(0.2);

	TFile * file_z1000 = new TFile("plots/MC_root/zprime1000.el_dis4_hist.root");
	TH1F * h_z1000 = (TH1F*)file_z1000->Get("m_event");
	h_z1000->Scale(0.55);

	TFile * file_z1250 = new TFile("plots/MC_root/zprime1250.el_dis4_hist.root");
	TH1F * h_z1250 = (TH1F*)file_z1250->Get("m_event");
	h_z1250->Scale(0.19);

	TFile * file_z1500 = new TFile("plots/MC_root/zprime1500.el_dis4_hist.root");
	TH1F * h_z1500 = (TH1F*)file_z1500->Get("m_event");
	h_z1500->Scale(0.083);

	TFile * file_z1750 = new TFile("plots/MC_root/zprime1750.el_dis4_hist.root");
	TH1F * h_z1750 = (TH1F*)file_z1750->Get("m_event");
	h_z1750->Scale(0.03);

	TFile * file_z2000 = new TFile("plots/MC_root/zprime2000.el_dis4_hist.root");
	TH1F * h_z2000 = (TH1F*)file_z2000->Get("m_event");
	h_z2000->Scale(0.014);

	TFile * file_z2250 = new TFile("plots/MC_root/zprime2250.el_dis4_hist.root");
	TH1F * h_z2250 = (TH1F*)file_z2250->Get("m_event");
	h_z2250->Scale(0.00067);

	TFile * file_z2500 = new TFile("plots/MC_root/zprime2500.el_dis4_hist.root");
	TH1F * h_z2500 = (TH1F*)file_z2500->Get("m_event");
	h_z2500->Scale(0.00035);

	TFile * file_z3000 = new TFile("plots/MC_root/zprime3000.el_dis4_hist.root");
	TH1F * h_z3000 = (TH1F*)file_z3000->Get("m_event");
	h_z3000->Scale(0.00012);

	float limitXsec[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	// chi squared 66.339 for exclusion limit
	float chi = 100; 
	float sc = 1;
	int n = 0;
	float chi_lim = 37.652;//14.611;//37.652;

	// test if x-section to large or to small
	cout << "test z400" << endl;
	TH1F * h_background = (TH1F*)h_MC->Clone();
	TH1F * h_signal = (TH1F*)h_z400->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z400->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z400->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[0] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;



// test if x-section to large or to small
	cout << "test z500" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z500->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z500->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z500->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[1] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;



// test if x-section to large or to small
	cout << "test z750" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z750->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z750->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z750->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[2] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;



// test if x-section to large or to small
	cout << "test z1000" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z1000->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z1000->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z1000->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[3] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;



// test if x-section to large or to small
	cout << "test z1250" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z1250->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z1250->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z1250->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[4] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;



// test if x-section to large or to small
	cout << "test z1500" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z1500->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z1500->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z1500->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[5] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;


// test if x-section to large or to small
	cout << "test z1750" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z1750->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z1750->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z1750->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[6] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;


// test if x-section to large or to small
	cout << "test z2000" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z2000->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z2000->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z2000->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[7] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;


// test if x-section to large or to small
	cout << "test z2250" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z2250->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z2250->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z2250->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[8] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;


// test if x-section to large or to small
	cout << "test z2500" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z2500->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z2500->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z2500->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[9] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;


// test if x-section to large or to small
	cout << "test z3000" << endl;
	h_background = (TH1F*)h_MC->Clone();
	h_signal = (TH1F*)h_z3000->Clone();
	h_background->Add(h_signal);
	chi = chisquareNBins(h_data, h_background);
	sc = 1;
	delete h_background;
	delete h_signal;

	if (chi < chi_lim){
		cout << "kleiner " << chi << endl;
		do{
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z3000->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += 0.01; 
		n++;

		delete h_background;
		delete h_signal;
		} while (chi < chi_lim && n < 10000);

	}else{
		cout << "größer " << chi << endl;
	while(chi > chi_lim && n < 10000) {
		TH1F * h_background = (TH1F*)h_MC->Clone();
		TH1F * h_signal = (TH1F*)h_z3000->Clone();
		h_signal->Scale(sc);
		h_background->Add(h_signal);

		chi = chisquareNBins(h_data, h_background);
		sc += -0.01; 
		n++;

		delete h_background;
		delete h_signal;
	}

	}

	// save the limit value of sigma
	limitXsec[10] = sc; 
	cout << "Chi " << chi << " in " << n << " mit " << sc << endl;



















	float N_MC[11] = {100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000}; // * 100000;
	//N_MC = N_MC * float(100000);
	float lumi = 1 * pow(10, 3);
	float weights[11] = {1.10000000e+00, 8.20000000e-01, 2.00000000e-01, 5.50000000e-02, 1.90000000e-02, 8.30000000e-03, 3.00000000e-03, 1.40000000e-03, 6.70000000e-04,  3.50000000e-04, 1.20000000e-04};

	for (int i = 0; i < 11; i++){
		limitXsec[i] = limitXsec[i] * weights[i] * N_MC[i] / 	lumi;
	}



	// Once you finished calculating your limits, you want to plot them
	// Start with arrays of the mass, the expected cross section and the limits you calculated
	// The mass array has already be filled for you in order to remind you of the use of arrays
	float mass[11] = {400.,500.,750.,1000.,1250.,1500.,1750.,2000.,2250.,2500.,3000};
	float expectedXsec[11] = {1.1e2, 8.2e1, 2.0e1, 5.5, 1.9, 8.3e-1, 3.0e-1, 1.4e-1, 6.7e-2, 3.5e-2, 1.2e-2};
//	float limitXsec[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	// Create a canvas
	SetStyle();
	TCanvas * c_limits = new TCanvas("c_limits", "canvas for limit plot", 1);
	// If you want to, use a logarithmic y axis 
  	c_limits->SetLogy();

	// Create a TGraph for the expected cross section
	TGraph * g_expected = new TGraph(11, mass, expectedXsec);
	g_expected->SetLineColor(kBlue);

	//Create a TGraph with you limits
	TGraph * g_limits = new TGraph(11, mass, limitXsec);
	
	//The TH1D is only to have axes to you plot
	TH1D * h_helper = new TH1D("h_helper", "just an empty helper histogram", 1, 400., 3000.);
	h_helper->SetMaximum(270);
	h_helper->GetXaxis()->SetTitle("m_{Z\'} [GeV]"); 
  	h_helper->GetYaxis()->SetTitle("#sigma_{Z'}#timesBR(Z'#rightarrow t#bar{t}) [pb]"); // don't forget the axis titles !
  	h_helper->Draw("p");

	// create a legend
	TLegend * l = new TLegend(0.35, 0.7, 0.9, 0.8, "");
  	l->SetFillColor(0);
  	l->SetBorderSize(0);
  	l->SetTextSize(0.04);
  	l->SetTextAlign(12);
  	l->AddEntry(g_expected, "Expected #sigma_{Z'}#timesBR(Z'#rightarrow t#bar{t})", "l");
  	l->AddEntry(g_limits, "Observed 95% CL upper limit (100 pb^{-1})", "l");
  	
  	g_expected->Draw("l SAME"); 
  	g_limits->Draw("l SAME");
  	l->Draw();
  	c_limits->SetLogy();
  	c_limits->Print("limits.pdf");
	
	for(int i = 0; i<11; i++){
	cout << limitXsec[i] << endl;
	}
	cout << limitXsec << endl;

	return 0;

}


float chisquareNBins(TH1F * data, TH1F * mc){

	float chisquare_test = 0.0;

	int nbins = data->GetSize();
	int nbinsused = 0;
	for(int i = 1; i < nbins; i++){
		float n_data = data->GetBinContent(i);
		float n_mc = mc->GetBinContent(i);
		if(n_mc != 0.){
			chisquare_test = chisquare_test + (n_data-n_mc)*(n_data-n_mc)/n_mc;
			nbinsused++;
		}
	}


	double ndf= double(nbinsused-1);

//	cout << "The number of degrees of freedom is " << ndf << endl;
	return chisquare_test;

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
