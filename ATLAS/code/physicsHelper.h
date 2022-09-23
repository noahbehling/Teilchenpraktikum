#ifndef physicsHelper_h
#define physicsHelper_h

#include "NeutrinoReco.cc"


class physicsHelper {

  public:
    static TLorentzVector * Neutrino(TLorentzVector metvect, TLorentzVector lepton) {

	  	double* pz  = new double[2];
  		int nsol = pz_of_W(lepton, &metvect, pz);
                TLorentzVector* neutrino = new TLorentzVector();
  		double nuE = sqrt(metvect.Px()*metvect.Px() + metvect.Py()*metvect.Py() + pz[0]*pz[0]);
  		neutrino->SetPxPyPzE(metvect.Px(),metvect.Py(),pz[0], nuE);
  		delete pz;
  		if(nsol > 2) {
    		std::cout << "Warning: nsol should not be larger than 2, nsol = " << nsol << std::endl;
    		return NULL;
  		}
  		return neutrino;
		}

};

#endif
