/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

// ====================================================================
//
// Description:
//
// This module implements an analysis that tries to infer to which kind
// of memory (global, local, shared, const, param) each pointer in the 
// code points to. It uses a global forward flow analysis to propagate
// information in the program representation in order to determine more
// precisely to which memory space each pointer in each program point
// points to.
//
//
// Usage:
//   
//   PTR_CLASS *p = new PTR_CLASS(); // create new analysis object
//   p->Compute(pu_wn);              // do the analysis
//
//   For any ILOAD/ISTORE:           
//   PtrClasses pclass = p->Query(wn); // get the ptr space addressed.
//                                     // by this ILOAD/ISTORE
// ====================================================================

#ifndef opt_ptrclass_INCLUDED
#define opt_ptrclass_INCLUDED

#include <list>
#include <map>

#include "wn.h"                     /* Whirl node def */
#include "mempool.h"                /* for MEM_POOL */
#include "wn_util.h"                /* iterators */
#include "wn_tree_util.h"           /* postfix iterators */
#include "tracing.h"		    /* for Get_Trace() */
#include "opt_defs.h"               /* for trace point defs */

using namespace std;

enum PtrClasses {
  Top=0,
  Bottom,
  Global,
  Local,
  Constant,
  Shared,
  Param,
  Texture,
  Generic,
};

enum PtrClasses MergePtrClasses(enum PtrClasses a, enum PtrClasses b) ;
char GetPtrClassesName(enum PtrClasses a);

// a unique "location" is symbol+offset.
typedef struct loctag
{
    ST *sym;
    WN_OFFSET off;
    loctag() {sym = 0; off = 0;}
    loctag(ST *s, WN_OFFSET o) { sym = s; off = o; }
    void Set(ST *s, WN_OFFSET off_in) { sym = s; off = off_in; };
}loc_t;

// function for comparing locations.
struct loc_t_cmp
{
    bool operator() (const loc_t a, const loc_t b)
    {
        if (a.sym < b.sym || (a.sym == b.sym && (a.off < b.off)))
            return true;
        return false;
    };
};

// map from location --> ptrclass.
typedef std::map<loc_t, PtrClasses, loc_t_cmp> locmap_t;


// Pstate: State object for ptr classification analysis
class Pstate 
{
private:
    // map: location --> ptrclass.
    locmap_t *locmap;
    friend class PTR_CLASS;
    void WalkType(ST *root_st, TY_IDX ty, WN_OFFSET off, PtrClasses pstate);

public:
    Pstate(void) { locmap = new locmap_t(); };
    ~Pstate(void){ delete locmap; };

    //--- Interface functions expected by DFSolve
    void CopyFrom(Pstate *in){  locmap->clear(); *locmap = *(in->locmap); };
    BOOL Merge(Pstate *in);
    void Empty(void) { locmap->clear(); };
    //---- end of interface functions.

    void Initialize(WN *pu);
    void ResetToBottom(void);
};

// PTR_CLASS: data flow analyzer for deciding what memory
// space an ILOAD/ISTORE can point to.
class PTR_CLASS 
{
    private:
        MEM_POOL _ptr_pool;

        // Map from WN* -> PtrClasses. We take advantage
        // of the fact that WN_MAP_32_Get returns "0" 
        // if not found in the map by defining enum PtrClasses::Top
        // to also be 0.
        WN_MAP _expr_map;           

        // Update expr map with new state. return true if 
        // new state different than existing state
        BOOL UpdateExprMap(WN *wn, PtrClasses pstate);

        BOOL _applied; // apply has been called
    public:
        //// Interface functions expected by DFSolve class
        void TransferFunction(WN *wn, Pstate *state);
        void Apply(WN *wn, Pstate *state);
        //// end of interface functions

        PTR_CLASS(void);
        ~PTR_CLASS(void);

        //// API functions for users
        // Do the pointer space analysis on the given function.
        void Compute(WN *pu); 
        // Query ptrclasses enum state for address computation of ILOAD/ISTORE
        enum PtrClasses Query(WN *inp);
        // whether all iload/istore uses can be resolved
        BOOL AllCanBeResolved;
        //// end of API functions
};
#endif

