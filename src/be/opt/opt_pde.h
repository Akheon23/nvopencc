/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

//  
//  WOPT pde options :
//  
//  -WOPT:wopt_pde=1 
//      enables WOPT pde
//  
//  -WOPT:pde_max_iters=<n>
//      n=0 runs the pde until no more operations can be moved.
//      n!=0 runs exactly 'n' number of iterations.
//  

#ifndef opt_pde_INCLUDED
#define opt_pde_INCLUDED

#include "defs.h"
#include "errors.h"
#include "erglob.h"
#include "glob.h" // for Cur_PU_Name
#include "tracing.h"
#include "pf_cg.h"

#include "cxx_base.h"
#include "erbe.h"

#include "opt_base.h"
#include "opt_bb.h"
#include "opt_config.h"
#include "opt_sys.h"
#include "bb_node_set.h"
#include "idx_32_set.h"
#include "opt_cfg.h"
#include "opt_ssa.h"
#include "opt_du.h"

#include "opt_sym.h"
#include "opt_mu_chi.h"
#include "opt_htable.h"
#include "opt_main.h"
#include "opt_alias_rule.h"
#include "opt_exc.h"
#include "opt_util.h"
#include "opt_project.h"
#include "opt_rvi.h"
#include "opt_rviwn.h"
#include "opt_ptrclass.h"
#include <vector>
#include <set>
#include <map>

#include <ext/hash_map>

#include "wn_dfsolve.h"
#include "ir_reader.h"

using namespace std;
using namespace __gnu_cxx;

//  eqWN is a functor which compares two <WN *>s. That's
//  how hash_map needs the "equal to" function, namely, a
//  class with the operator() overloaded to compare
//  two objects. Comparing two <WN *>s is easy, just
//  return (wn1 == wn2).
//  
struct eqWN {
  bool operator()(WN *wn1, WN *wn2) const {
    return (wn1 == wn2);
  }
};

//  Below is the hash function for WN*. hash_map again needs
//  a functor for the job, basically a class with operator()
//  overloaded to do the hashing. Here, the class hash<T> is
//  being specialized for <WN *>, hence the strange (to some people)
//  syntax template<>.
//  The hash function used here is WN_map_id(wn), which is the
//  same function used by WN_MAP. It returns a unique value for
//  every wn*.
//  

namespace __gnu_cxx {

template<> struct hash<WN *> {
  size_t operator() (WN *wn) const {
    return WN_map_id(wn);
  }
};

}

//  Data flow solver for delayability.
//  The data flow information tracked is the delayed set, represented
//  using IDX_32_SET. The Apply function populates the delayed_map, 
//  declared in opt_pde.cxx.
//  This is a forward analysis.
//  
struct DelayabilityState {

  IDX_32_SET *delayed;

  DelayabilityState();
  ~DelayabilityState();

  void CopyFrom(DelayabilityState *);
  void Empty();
  BOOL Merge(DelayabilityState *);
};

struct DelayabilityClient {

  void TransferFunction(WN *, DelayabilityState *);
  void Apply(WN *, DelayabilityState *);
};

#if 0
// Liveness related types

// Liveness is tracked for variables. A variable is represented by
// a unique location which is is <sym, offset> pair.
typedef struct loctag {
  ST *sym;
  WN_OFFSET offset;

  loctag(ST *s, WN_OFFSET o) { sym = s; offset = o; }
} loc_t;

// Comparator for loc_t
struct loc_t_cmp {
  bool operator() (const loc_t a, const loc_t b) const
  {
    if((a.sym < b.sym) ||
       ((a.sym == b.sym) && (a.offset < b.offset)))
      return true;
    return false;
  }
};
#endif
typedef set<loc_t, loc_t_cmp> loc_set_t;

// Class to hold the current liveness information
struct LivenessState {
  loc_set_t _cur_live_vars;

  LivenessState();
  ~LivenessState();

  void CopyFrom(LivenessState *);
  BOOL Merge(LivenessState *);
  void Empty(void);

  void Print(void);
};

// Data flow client for liveness
struct LivenessClient {

  void TransferFunction(WN *, LivenessState *);
  void Apply(WN *, LivenessState *);
};

WN *Perform_PDE(WN *, ALIAS_MANAGER *);

#endif
