/////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Mar 26 15:40:36 2015 by ROOT version 5.34/18
// from TTree mini/4-vectors + variables required for scaling factors
// found on file: el.root
//////////////////////////////////////////////////////////

#ifndef mini_h
#define mini_h

#include <TROOT.h>
#include <TChain.h>
#include <TString.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

// Fixed size dimensions of array or collections stored in the TTree if any.

class mini {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Int_t       runNumber                    ;
   Int_t       eventNumber                  ;
   Int_t       channelNumber                ;
   Int_t       lbNumber                     ;
   Int_t       rndRunNumber                 ;
   Float_t     mu                           ;
   Float_t     mcWeight                     ;
   Float_t     eventWeight_PRETAG           ;
   Float_t     eventWeight_BTAG             ;
   Int_t       pvxp_n                       ;
   Float_t     vxp_z                        ;
   Float_t     scaleFactor_PILEUP           ;
   Float_t     scaleFactor_ELE              ;
   Float_t     scaleFactor_MUON             ;
   Float_t     scaleFactor_BTAG             ;
   Float_t     scaleFactor_TRIGGER          ;
   Float_t     scaleFactor_WJETSNORM        ;
   Float_t     scaleFactor_WJETSSHAPE       ;
   Float_t     scaleFactor_JVFSF            ;
   Float_t     scaleFactor_ZVERTEX          ;
   Float_t     scaleFactor_ALLPRETAG        ;
   Float_t     scaleFactor_ALLBTAG          ;
   Float_t     scaleFactor_COMBINED         ; 
   Bool_t      trigE                        ;
   Bool_t      trigM                        ;
   Bool_t      passGRL                      ;
   Bool_t      hasGoodVertex                ;
   Float_t     scaledWeight                 ;
   UInt_t      lep_n                        ;
   Bool_t      lep_truthMatched[100]        ;
   UShort_t    lep_trigMatched[100]         ;
   Float_t     lep_pt[100]                  ;
   Float_t     lep_eta[100]                 ;
   Float_t     lep_phi[100]                 ;
   Float_t     lep_E[100]                   ;
   Float_t     lep_z0[100]                  ;
   Float_t     lep_charge[100]              ;
   Bool_t      lep_isTight[100]             ;
   UInt_t      lep_type[100]                ;
   UInt_t      lep_flag[100]                ;
   Float_t     el_cl_eta[100]               ;
   Float_t     lep_ptcone30[100]            ;
   Float_t     lep_etcone20[100]            ;
   Float_t     lep_miniIso10_4[100]         ;
   Float_t     lep_trackd0pvunbiased[100]   ;
   Float_t     lep_tracksigd0pvunbiased[100];
   Float_t     met_et                       ;
   Float_t     met_phi                      ;
   UInt_t      jet_n                        ;
   UInt_t      alljet_n                     ;
   Float_t     jet_pt[100]                  ;
   Float_t     jet_eta[100]                 ;
   Float_t     jet_phi[100]                 ;
   Float_t     jet_E[100]                   ;
   Float_t     jet_m[100]                   ;
   Float_t     jet_jvf[100]                 ;
   Int_t       jet_trueflav[100]            ;
   Int_t       jet_truthMatched[100]        ;
   Float_t     jet_SV0[100]                 ;
   Float_t     jet_MV1[100]                 ;
   UInt_t      jet_flag[100]                ;
   Bool_t      jet_BadMediumBCH[100]        ;
   Bool_t      jet_BadTightBCH[100]         ;
   Float_t     jet_emfrac[100]              ;
   Float_t     jet_BCH_CORR_CELL[100]       ;
   Bool_t      jet_good[100]                ; 

   // List of branches

   TBranch  *b_runNumber                    ;
   TBranch  *b_eventNumber                  ;
   TBranch  *b_channelNumber                ;
   TBranch  *b_lbNumber                     ;
   TBranch  *b_rndRunNumber                 ;
   TBranch  *b_mu                           ;
   TBranch  *b_mcWeight                     ;
   TBranch  *b_eventWeight_PRETAG           ;
   TBranch  *b_eventWeight_BTAG             ;
   TBranch  *b_pvxp_n                       ;
   TBranch  *b_vxp_z                        ;
   TBranch  *b_scaleFactor_PILEUP           ;
   TBranch  *b_scaleFactor_ELE              ;
   TBranch  *b_scaleFactor_MUON             ;
   TBranch  *b_scaleFactor_BTAG             ;
   TBranch  *b_scaleFactor_TRIGGER          ;
   TBranch  *b_scaleFactor_WJETSNORM        ;
   TBranch  *b_scaleFactor_WJETSSHAPE       ;
   TBranch  *b_scaleFactor_JVFSF            ;
   TBranch  *b_scaleFactor_ZVERTEX          ;
   TBranch  *b_scaleFactor_ALLPRETAG        ;
   TBranch  *b_scaleFactor_ALLBTAG          ;
   TBranch  *b_scaleFactor_COMBINED         ;
   TBranch  *b_trigE                        ;
   TBranch  *b_trigM                        ;
   TBranch  *b_passGRL                      ;
   TBranch  *b_hasGoodVertex                ;
   TBranch  *b_scaledWeight                 ;
   TBranch  *b_lep_n                        ;
   TBranch  *b_lep_truthMatched             ;
   TBranch  *b_lep_trigMatched              ;
   TBranch  *b_lep_pt                       ;
   TBranch  *b_lep_eta                      ;
   TBranch  *b_lep_phi                      ;
   TBranch  *b_lep_E                        ;
   TBranch  *b_lep_z0                       ;
   TBranch  *b_lep_charge                   ;
   TBranch  *b_lep_isTight                  ;
   TBranch  *b_lep_type                     ;
   TBranch  *b_lep_flag                     ;
   TBranch  *b_el_cl_eta                    ;
   TBranch  *b_lep_ptcone30                 ;
   TBranch  *b_lep_etcone20                 ;
   TBranch  *b_lep_miniIso10_4              ;
   TBranch  *b_lep_trackd0pvunbiased        ;
   TBranch  *b_lep_tracksigd0pvunbiased     ;
   TBranch  *b_met_et                       ;
   TBranch  *b_met_phi                      ;
   TBranch  *b_jet_n                        ;
   TBranch  *b_alljet_n                     ;
   TBranch  *b_jet_pt                       ;
   TBranch  *b_jet_eta                      ;
   TBranch  *b_jet_phi                      ;
   TBranch  *b_jet_E                        ;
   TBranch  *b_jet_m                        ;
   TBranch  *b_jet_jvf                      ;
   TBranch  *b_jet_trueflav                 ;
   TBranch  *b_jet_truthMatched             ;
   TBranch  *b_jet_SV0                      ;
   TBranch  *b_jet_MV1                      ;
   TBranch  *b_jet_flag                     ;
   TBranch  *b_jet_BadMediumBCH             ;
   TBranch  *b_jet_BadTightBCH              ;
   TBranch  *b_jet_emfrac                   ;
   TBranch  *b_jet_BCH_CORR_CELL            ;
   TBranch  *b_jet_good                     ;


   mini(TTree *tree=0);
   virtual ~mini();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Int_t    GetEntries() { return fChain->GetEntries(); }
   virtual Int_t    GetEntries(const char* sel) { return fChain->GetEntries(sel); }
   virtual TTree *  CloneTree() { return fChain->CloneTree(0); }
   virtual TTree *  CloneTree(int a) { return fChain->CloneTree(a); }
   virtual void     CopyAddresses(TTree *tree){ fChain->CopyAddresses(tree, true);}
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
   virtual Double_t GetMinimum(TString varName);
   virtual Double_t GetMaximum(TString varName);

};

#endif

#ifdef mini_cxx
mini::mini(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("el.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("el.root");
      }
      f->GetObject("mini",tree);

   }
   Init(tree);
}

mini::~mini()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t mini::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t mini::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void mini::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);


   fChain->SetBranchAddress("runNumber"                ,&runNumber                    ,&b_runNumber                    );
   fChain->SetBranchAddress("eventNumber"              ,&eventNumber                  ,&b_eventNumber                  );
   fChain->SetBranchAddress("channelNumber"            ,&channelNumber                ,&b_channelNumber                );
   fChain->SetBranchAddress("lbNumber"                 ,&lbNumber                     ,&b_lbNumber                     );
   fChain->SetBranchAddress("rndRunNumber"             ,&rndRunNumber                 ,&b_rndRunNumber                 );
   fChain->SetBranchAddress("mu"                       ,&mu                           ,&b_mu                           );
   fChain->SetBranchAddress("mcWeight"                 ,&mcWeight                     ,&b_mcWeight                     );
   fChain->SetBranchAddress("eventWeight_PRETAG"       ,&eventWeight_PRETAG           ,&b_eventWeight_PRETAG           );
   fChain->SetBranchAddress("eventWeight_BTAG"         ,&eventWeight_BTAG             ,&b_eventWeight_BTAG             );
   fChain->SetBranchAddress("pvxp_n"                   ,&pvxp_n                       ,&b_pvxp_n                       );
   fChain->SetBranchAddress("vxp_z"                    ,&vxp_z                        ,&b_vxp_z                        );
   fChain->SetBranchAddress("scaleFactor_PILEUP"       ,&scaleFactor_PILEUP           ,&b_scaleFactor_PILEUP           );
   fChain->SetBranchAddress("scaleFactor_ELE"          ,&scaleFactor_ELE              ,&b_scaleFactor_ELE              );
   fChain->SetBranchAddress("scaleFactor_MUON"         ,&scaleFactor_MUON             ,&b_scaleFactor_MUON             );
   fChain->SetBranchAddress("scaleFactor_BTAG"         ,&scaleFactor_BTAG             ,&b_scaleFactor_BTAG             );
   fChain->SetBranchAddress("scaleFactor_TRIGGER"      ,&scaleFactor_TRIGGER          ,&b_scaleFactor_TRIGGER          );
   fChain->SetBranchAddress("scaleFactor_WJETSNORM"    ,&scaleFactor_WJETSNORM        ,&b_scaleFactor_WJETSNORM        );
   fChain->SetBranchAddress("scaleFactor_WJETSSHAPE"   ,&scaleFactor_WJETSSHAPE       ,&b_scaleFactor_WJETSSHAPE       );
   fChain->SetBranchAddress("scaleFactor_JVFSF"        ,&scaleFactor_JVFSF            ,&b_scaleFactor_JVFSF            );
   fChain->SetBranchAddress("scaleFactor_ZVERTEX"      ,&scaleFactor_ZVERTEX          ,&b_scaleFactor_ZVERTEX          );
   fChain->SetBranchAddress("scaleFactor_ALLPRETAG"    ,&scaleFactor_ALLPRETAG        ,&b_scaleFactor_ALLPRETAG        );
   fChain->SetBranchAddress("scaleFactor_ALLBTAG"      ,&scaleFactor_ALLBTAG          ,&b_scaleFactor_ALLBTAG          );
   fChain->SetBranchAddress("scaleFactor_COMBINED"     ,&scaleFactor_COMBINED         ,&b_scaleFactor_COMBINED         );
   fChain->SetBranchAddress("trigE"                    ,&trigE                        ,&b_trigE                        );
   fChain->SetBranchAddress("trigM"                    ,&trigM                        ,&b_trigM                        );
   fChain->SetBranchAddress("passGRL"                  ,&passGRL                      ,&b_passGRL                      );
   fChain->SetBranchAddress("hasGoodVertex"            ,&hasGoodVertex                ,&b_hasGoodVertex                );
   fChain->SetBranchAddress("scaledWeight"             ,&scaledWeight                 ,&b_scaledWeight                 );
   fChain->SetBranchAddress("lep_n"                    ,&lep_n                        ,&b_lep_n                        );
   fChain->SetBranchAddress("lep_truthMatched"         , lep_truthMatched             ,&b_lep_truthMatched             );
   fChain->SetBranchAddress("lep_trigMatched"          , lep_trigMatched              ,&b_lep_trigMatched              );
   fChain->SetBranchAddress("lep_pt"                   , lep_pt                       ,&b_lep_pt                       );
   fChain->SetBranchAddress("lep_eta"                  , lep_eta                      ,&b_lep_eta                      );
   fChain->SetBranchAddress("lep_phi"                  , lep_phi                      ,&b_lep_phi                      );
   fChain->SetBranchAddress("lep_E"                    , lep_E                        ,&b_lep_E                        );
   fChain->SetBranchAddress("lep_z0"                   , lep_z0                       ,&b_lep_z0                       );
   fChain->SetBranchAddress("lep_charge"               , lep_charge                   ,&b_lep_charge                   );
   fChain->SetBranchAddress("lep_isTight"              , lep_isTight                  ,&b_lep_isTight                  );
   fChain->SetBranchAddress("lep_type"                 , lep_type                     ,&b_lep_type                     );
   fChain->SetBranchAddress("lep_flag"                 , lep_flag                     ,&b_lep_flag                     );
   fChain->SetBranchAddress("el_cl_eta"                , el_cl_eta                    ,&b_el_cl_eta                    );
   fChain->SetBranchAddress("lep_ptcone30"             , lep_ptcone30                 ,&b_lep_ptcone30                 );
   fChain->SetBranchAddress("lep_etcone20"             , lep_etcone20                 ,&b_lep_etcone20                 );
   fChain->SetBranchAddress("lep_miniIso10_4"          , lep_miniIso10_4              ,&b_lep_miniIso10_4              );
   fChain->SetBranchAddress("lep_trackd0pvunbiased"    , lep_trackd0pvunbiased        ,&b_lep_trackd0pvunbiased        );
   fChain->SetBranchAddress("lep_tracksigd0pvunbiased" , lep_tracksigd0pvunbiased     ,&b_lep_tracksigd0pvunbiased     );
   fChain->SetBranchAddress("met_et"                   ,&met_et                       ,&b_met_et                       );
   fChain->SetBranchAddress("met_phi"                  ,&met_phi                      ,&b_met_phi                      );
   fChain->SetBranchAddress("jet_n"                    ,&jet_n                        ,&b_jet_n                        );
   fChain->SetBranchAddress("alljet_n"                 ,&alljet_n                     ,&b_alljet_n                     );
   fChain->SetBranchAddress("jet_pt"                   , jet_pt                       ,&b_jet_pt                       );
   fChain->SetBranchAddress("jet_eta"                  , jet_eta                      ,&b_jet_eta                      );
   fChain->SetBranchAddress("jet_phi"                  , jet_phi                      ,&b_jet_phi                      );
   fChain->SetBranchAddress("jet_E"                    , jet_E                        ,&b_jet_E                        );
   fChain->SetBranchAddress("jet_m"                    , jet_m                        ,&b_jet_m                        );
   fChain->SetBranchAddress("jet_jvf"                  , jet_jvf                      ,&b_jet_jvf                      );
   fChain->SetBranchAddress("jet_trueflav"             , jet_trueflav                 ,&b_jet_trueflav                 );
   fChain->SetBranchAddress("jet_truthMatched"         , jet_truthMatched             ,&b_jet_truthMatched             );
   fChain->SetBranchAddress("jet_SV0"                  , jet_SV0                      ,&b_jet_SV0                      );
   fChain->SetBranchAddress("jet_MV1"                  , jet_MV1                      ,&b_jet_MV1                      );
   fChain->SetBranchAddress("jet_flag"                 , jet_flag                     ,&b_jet_flag                     );
   fChain->SetBranchAddress("jet_BadMediumBCH"         , jet_BadMediumBCH             ,&b_jet_BadMediumBCH             );
   fChain->SetBranchAddress("jet_BadTightBCH"          , jet_BadTightBCH              ,&b_jet_BadTightBCH              );
   fChain->SetBranchAddress("jet_emfrac"               , jet_emfrac                   ,&b_jet_emfrac                   );
   fChain->SetBranchAddress("jet_BCH_CORR_CELL"        , jet_BCH_CORR_CELL            ,&b_jet_BCH_CORR_CELL            );
   fChain->SetBranchAddress("jet_good"                 , jet_good                     ,&b_jet_good                     );


   Notify();
}

Bool_t mini::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void mini::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t mini::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}


Double_t mini::GetMinimum(TString varName)
{
   if (!fChain) return 0.0;
   return fChain->GetMinimum(varName);
}

Double_t mini::GetMaximum(TString varName)
{
   if (!fChain) return 0.0;
   return fChain->GetMaximum(varName);
}



#endif // #ifdef mini_cxx
