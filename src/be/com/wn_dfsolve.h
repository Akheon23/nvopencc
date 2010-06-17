/*
 * Copyright 2009-2010 NVIDIA Corporation.  All rights reserved.
 */
#ifndef wn_dfsolve_INCLUDED
#define wn_dfsolve_INCLUDED

#include "wn.h"
#include "mempool.h"
#include "wn_util.h"
#include "wn_tree_util.h"
#include "erbe.h"

#include <list>
#include <map>
#include <ext/hash_map>

// ====================================================================
// DFSolver: An iterative data flow solver that operates on WHIRL IR. 
//
// Description:
// 1. The DFSolver is parameterized by 2 classes - TClient and TState.  
// 2. TState represents dataflow state that can be different at different
// points in the program. TClient represents the particular data flow analysis
// being implemented.  DFSolver expects TState, TClient to implement certain
// interface functions (described below). Apart from these functions, TClient & TState
// are opaque to DFSolver.
// 3. The TState class provides methods to manipulate dataflow state. It must
// provide the following methods:
// 
//      void CopyFrom(TState *in) : Set state to the "*in".
//      BOOL Merge(Pstate *in)    : Merge current state with "*in". Return TRUE if state
//                                  changed.
//      void Empty(void)          : Empty the state.
//
//
// 4. The TClient object provides methods to achieve a particular data flow analysis.
// It must provide the following methods:
//
//      void TransferFunction(WN *wn, TState *state) : Update "*state" with the effect
//                                                     of the WHIRL node wn.
//      void Apply(WN *wn, TState *state)            : Apply "*state" for whirl node wn.
//
//
// 5. DFSolver.Solve() invokes the iterative data flow solver. The client's 
// TransferFunction() is called to model the effect of a whirl node on the current
// state. (current state is always "_state" in the code below). Once fixed point
// has been reached, the WHIRL tree is iterated one last time in "APPLY" mode. 
// In APPLY mode, the client's Apply()  is invoked to allow it to use the
// results of the data flow analysis.
//
// Note: Clients should not change the structure of the WHIRL tree while DFSolver is
// active. Instead, the client can save summary information inside Apply() and then
// run a separate pass that uses this information to modify the IR.
//
//
// be/opt/opt_ptrclass has an example data flow analysis pass written with this
// solver:
//    PTR_CLASS *client = new PTR_CLASS();          // Client
//      Pstate *mystate = new Pstate();             // Initial state object I
//    mystate->Initialize(tree);                    // Initialize I with info.
//    DFSolver<PTR_CLASS, Pstate> Solver(client);   // Create DFSolver object
//    Solver.Solve(TRUE, tree, mystate);            // Solve!
//    
//
//
//  ====================================================================
template <class TClient, class TState> class DFSolver
{
    typedef __gnu_cxx::hash_map<INT32, WN *> label_map_t;
    private:
        // all class scope variables are always prefixed by "_"
        TClient *_client;                     // client implements the data flow analysis
        TState *_state;                       // current dataflow state
        MEM_POOL _pool;                       // pool for temporary allocations
        WN_MAP _state_map;                    // (WN*) -> (TState*) mapping for certain nodes
        label_map_t _label_map;               // label_id -> (WN*) for OPR_LABEL nodes.
        BOOL _change;                         // if TRUE, indicates fixed point not reached
        BOOL _apply;                          // if TRUE, indicates in APPLY mode (after fixed point)
        std::list<TState *>_alloced_states;   // for tracking temporary state allocations

    public:
        // ====================================================================
        //
        // Description: Constructor. Init with client data flow analyzer object.
        //
        // ====================================================================
        DFSolver(TClient *cl) 
        { 
            MEM_POOL_Initialize(&_pool, "df solver pool", FALSE);
            Init(cl); 
        }; // DFSolver
        
        // ====================================================================
        //
        // Description: Cache the point to the client data flow analysis object.
        //
        // ====================================================================
        void Init(TClient *cl)
        { 
            _client = cl; 
        }; // Init

        // ====================================================================
        //
        // Description: Destructor. Delete memory pool.
        //
        // ====================================================================
        ~DFSolver(void)
        {
            MEM_POOL_Delete(&_pool);
        }; // ~DFSolver

        // ====================================================================
        //
        // Description: Main function. Apply iterative data flow analysis over
        // the given whirl tree. Function Parameters:
        //   BOOL forward      : TRUE = forward, FALSE = backward data flow
        //                       analysis.
        //   WN *pu            : Root node of whirl tree. Expected to be a function
        //                       entry node.
        //   TState *init_state: Pointer to state object containing initial state
        //
        // The Solve() function does the following:
        // 1. Create "cached" versions of state for certain whirl nodes. For forward
        //    data flow, these are OPR_FUNC_ENTRY, OPR_LABEL and the structured
        //    control flow nodes. For backward data flow, these are OPR_RETURN,
        //    OPR_RETURN_VAL, OPR_LABEL and structured control flow nodes.
        // 2. Allocate memory for a "current state" object, pointed to by _state.
        // 3. Call Iterate to propagate data flow information.
        // 4. Free memory, including cached copies of the "state".
        //
        // Note: Not all structured control flow constructs are supported yet.
        // ====================================================================
        void Solve(BOOL forward, WN *pu, TState *init_state)
        {
            WN_ITER *itr;
            WN *wn;
            TState *s;
            typename std::list<TState *>::iterator liter;
            BOOL backward = !forward;

            FmtAssert( WN_operator(pu) == OPR_FUNC_ENTRY, ("Must pass func entry"));
            MEM_POOL_Push(&_pool);

            _apply = FALSE;
            _state = CreateState();
            
            // attach State nodes to required nodes
            _state_map = WN_MAP_Create(&_pool);

            for (itr = WN_WALK_TreeIter(pu); itr != NULL; itr = WN_WALK_TreeNext(itr)) {
                wn = WN_ITER_wn(itr);
                switch (WN_operator(wn)) {
                case OPR_LABEL:
                    _label_map[WN_label_number(wn)] = wn;
                    s = CreateState();
                    WN_MAP_Set(_state_map, wn, s);
                    break;
                case OPR_FUNC_ENTRY:
                    if (forward) {
                        s = CreateState();
                        WN_MAP_Set(_state_map, wn, s);
                        s->CopyFrom(init_state);
                    }
                    break;
                case OPR_IF:
                    if (forward) {
                        s = CreateState();
                        WN_MAP_Set(_state_map, wn, s);
                    }
                    break;
                case OPR_WHILE_DO:
                    if (forward) {
                        s = CreateState();
                        WN_MAP_Set(_state_map, wn, s);
                        s = CreateState();
                        WN_MAP_Set(_state_map, WN_while_test(wn), s);
                    }
                    break;
                case OPR_RETURN:
                case OPR_RETURN_VAL:
                    if (backward) {
                        s = CreateState();
                        WN_MAP_Set(_state_map, wn, s);
                        s->CopyFrom(init_state);
                    }
                    break;
                case OPR_TRUEBR:
                case OPR_FALSEBR:
                  if(backward) {
                        s = CreateState();
                        WN_MAP_Set(_state_map, wn, s);
                        s->Empty();
                  }
                  break;
                default: // add others here
                    break;
                }
            }
 
            Iterate(forward, pu);
            
            // free memory
            for (liter = _alloced_states.begin(); liter != _alloced_states.end(); liter++)
                delete *liter;

            _alloced_states.clear();
            _label_map.clear();
            WN_MAP_Delete(_state_map);
            MEM_POOL_Pop(&_pool);

        }; // Solve
    
    private:
        // ====================================================================
        //
        // Description: Allocate memory for a state object on the internal pool.
        // Also save the object pointer into "_alloced_states", used during 
        // cleanup to free allocated states.
        // ====================================================================
        TState *CreateState(void)
        {
            TState *s;
            s = new TState;
            s->Empty();
            _alloced_states.push_back(s);
            return s;
        }; // CreateState

        // ====================================================================
        //
        // Description: For a given LABEL node, return the cached "State" object
        // associated with it.
        // ====================================================================
        TState *GetLabelState(WN *wn)
        {
            WN *kid;
            label_map_t::iterator itr;
            INT32 label;
            
            label = WN_label_number(wn);
            itr = _label_map.find(label);
            FmtAssert((itr != _label_map.end()), ("label wn not found"));
            kid = itr->second;
            return (TState *)WN_MAP_Get(_state_map, kid);
        }; // GetLabelState

        // ====================================================================
        //
        // Description: Iterate in forward or backward directions while _change 
        // is TRUE. Once fixed point is reached (i.e. _change is FALSE), iterate
        // one last time in "Apply" mode.
        // ====================================================================
        void Iterate(BOOL forward, WN *pu_wn)
        {
            WN *wn;

            _change = TRUE;
            while (_change) {
                _change = FALSE;

                if (forward) {
                    TraverseForward(pu_wn);
                } else {
                    TraverseBackward(pu_wn);
                }
            }
            _apply = TRUE;
            if (forward)
                TraverseForward(pu_wn);
            else 
                TraverseBackward(pu_wn);
        }; // Iterate

        // ====================================================================
        //
        // Description: Walk the whirl tree in post-order (children before parents).
        // In APPLY mode, invoke the client's apply() function for the current
        // node being examined. Then invoke the transfer function for the current
        // node.
        // ====================================================================
        void TraverseForwardExpr(WN *root)
        {
            WN *wn;
            WN_TREE_ITER<POST_ORDER> itr(root);
#ifndef __GNU_BUG_WORKAROUND
            for (; itr != LAST_POST_ORDER_ITER; itr++) {
#else
            for (; itr != WN_TREE_ITER<POST_ORDER, WN*>(); itr++) {
#endif
                wn = itr.Wn();
                if(_apply)
                    _client->Apply(wn, _state);
                _client->TransferFunction(wn, _state);
            }
        }; // TraverseForwardExpr
     
        // ====================================================================
        //
        // Description: Traverse the given WHIRL-tree for forward data flow.
        // We start with the OPR_FUNC_ENTRY node. For each OPR_BLOCK node, we
        // walk its children in sequence. OPR_LABEL and control transfer statements
        // are handled specially. For all other statements S, call
        // TraverseForwardExpr to iterate over the children of the S in post_order.
        // 
        // _state always represents the current state. If the cached state of any
        // whirl node (e.g. associated with OPR_LABEL) changes, then _change is
        // set to TRUE (indicating that we need to iterate again).
        //
        // In APPLY mode, the client's apply function is always invoked for a
        // node BEFORE the transfer function for that node is invoked. 
        //
        // Note: currently we do not support DO_WHILE and DO_LOOP structured control
        // flow constructs. The implementation for WHILE_DO shows how these can
        // be implemented.
        //
        // Note: The TransferFunction gets called on the WHIRL nodes corresponding to
        // TRUEBR, FALSEBR, GOTO, RETURN, RETURN_VAL, LABEL. However, TransferFunction
        // is NOT called on code container nodes, like IF, WHILE_DO, FUNC_ENTRY, and
        // BLOCK.
        // ====================================================================
        void TraverseForward(WN *wn)
        {
            WN *kid, *next_wn;
            TState *s, *s_if, *s_then, *s_saved, *s_while_end, *s_while_start;
            label_map_t::iterator itr;
            OPERATOR opr;
            BOOL saved_apply;

            opr = WN_operator(wn);
            switch(opr) {
            case OPR_BLOCK:
                if (_apply)
                    _client->Apply(wn, _state);
                kid = WN_first(wn);
                while (kid) {
                    next_wn = WN_next(kid);
                    TraverseForward(kid);
                    kid = next_wn;
                };
                break;

            case OPR_REGION:
                if (_apply)
                    _client->Apply(wn, _state);
                TraverseForward(WN_kid2(wn));
                break;

            case OPR_FUNC_ENTRY:
                s = (TState *)WN_MAP_Get(_state_map, wn);
                _state->CopyFrom(s);
                if (_apply)
                    _client->Apply(wn, _state);
                TraverseForward(WN_func_body(wn));
                break;

            case OPR_IF:
                if (_apply)
                    _client->Apply(wn, _state);
                TraverseForwardExpr(WN_kid0(wn));
                if (WN_block_empty(WN_else(wn))) {
                    // slightly simpler
                    s_if = (TState *)WN_MAP_Get(_state_map, wn);
                    s_if->CopyFrom(_state);
                    TraverseForward(WN_then(wn));
                    _state->Merge(s_if);
                } else {
                    s_if = (TState *)WN_MAP_Get(_state_map, wn);
                    s_if->CopyFrom(_state);
                    TraverseForward(WN_then(wn));
                    s_then = _state;
                    _state = s_if;
                    TraverseForward(WN_else(wn));
                    s_then->Merge(_state);
                    _state = s_then; 
                }
                break;

            case OPR_WHILE_DO:
                if (_apply)
                    _client->Apply(wn, _state);
                s_while_end = (TState *)WN_MAP_Get(_state_map, wn);
                s_while_start = (TState *)WN_MAP_Get(_state_map, WN_while_test(wn));
                
                // The while-do can be thought of as this:
                //
                // label_while_start:
                //   bool B = Test_expr;
                //   if (!B) 
                //     goto label_while_end;
                //   while_body;
                //   goto label_while_start;
                //
                // label_while_end:
                //
                _change |= s_while_start->Merge(_state);
                if (_apply) 
                    FmtAssert(!_change, ("DF state changed in APPLY mode"));
                _state->CopyFrom(s_while_start);
                TraverseForward(WN_while_test(wn));
                _change |= s_while_end->Merge(_state);
                if (_apply) 
                    FmtAssert(!_change, ("DF state changed in APPLY mode"));
                TraverseForward(WN_while_body(wn));
                _change |= s_while_start->Merge(_state);
                if (_apply) 
                    FmtAssert(!_change, ("DF state changed in APPLY mode"));
                _state->CopyFrom(s_while_end);
                break;
              
            case OPR_DO_WHILE:
            case OPR_DO_LOOP:
                FmtAssert(0, ("Unhandled SCF!"));
                break;
    
            case OPR_LABEL:
                s = (TState *)WN_MAP_Get(_state_map, wn);
                _change |= s->Merge(_state);
                _state->CopyFrom(s);
                TraverseForwardExpr(wn);
                break;

            case OPR_TRUEBR:
            case OPR_FALSEBR:
                TraverseForwardExpr(wn);
                s = GetLabelState(wn);
                _change |= s->Merge(_state);
                if (_apply)
                    FmtAssert(!_change, ("DF state changed in APPLY mode"));
                break;

            case OPR_GOTO:                
            case OPR_REGION_EXIT:
                TraverseForwardExpr(wn);
                s = GetLabelState(wn);
                _change |= s->Merge(_state);
                if (_apply)
                    FmtAssert(!_change, ("DF state changed in APPLY mode"));
                _state->Empty();
                break;
    
            case OPR_RETURN:
            case OPR_RETURN_VAL:
                TraverseForwardExpr(wn);
                _state->Empty();
                break;

            case OPR_XGOTO:
            case OPR_AGOTO:
            case OPR_COMPGOTO:
                ErrMsgSrcpos(EC_Computed_GOTO, WN_Get_Linenum(wn));
                break;

            case OPR_IO:
            case OPR_GOTO_OUTER_BLOCK:
            case OPR_SWITCH:
            case OPR_CASEGOTO:
            case OPR_ALTENTRY:
                FmtAssert(0, ("Unhandled control flow"));
                break;
            default:
                FmtAssert(OPCODE_is_stmt(WN_opcode(wn)), ("expected statement"));
                TraverseForwardExpr(wn);
                break;
            }
        }; // TraverseForward

        // ====================================================================
        //
        // Description: Traverse the given WHIRL expression tree for backward 
        // data flow.
        // ====================================================================
        void TraverseBackwardExpr(WN *root) {
            WN *wn;
            WN_TREE_ITER<PRE_ORDER> itr(root);
#ifndef __GNU_BUG_WORKAROUND
            for (; itr != LAST_PRE_ORDER_ITER; itr++) {
#else
            for (; itr != WN_TREE_ITER<PRE_ORDER, WN*>(); itr++) {
#endif
                wn = itr.Wn();
                if(_apply)
                    _client->Apply(wn, _state);
                _client->TransferFunction(wn, _state);
            }
        }; // TraverseBackwardExpr

        
        // ====================================================================
        //
        // Description: Traverse the given WHIRL tree for backward data flow.
        // We start with the BLOCK node that is the child of the function entry.
        // For each block node, we start with the last statement in it, and 
        // walk backwards.
        //
        // OPR_LABEL, OPR_RETURN and OPR_RETURN_VAL cache state objects. If 
        // the state of a cached object changed, then _change is set to TRUE
        // to indicate that fixed point has not been reached.
        //
        // Note: This only supports MWHIRL for now (no structured control flow).
        // ====================================================================
        void TraverseBackward(WN *wn) {
            WN *kid, *prev_wn;
            TState *s, *s_if, *s_then, *s_saved;
            label_map_t::iterator itr;
            INT32 label;
            OPERATOR opr;

            opr = WN_operator(wn);
            switch(opr) {
            case OPR_BLOCK:
                kid = WN_last(wn);
                while (kid) {
                    prev_wn = WN_prev(kid);
                    TraverseBackward(kid);
                    kid = prev_wn;
                };
                if (_apply)
                    _client->Apply(wn, _state);
                break;

            case OPR_REGION:
                TraverseBackward(WN_kid2(wn));
                if (_apply)
                    _client->Apply(wn, _state);
                break;

            case OPR_RETURN:
            case OPR_RETURN_VAL:
                s = (TState *)WN_MAP_Get(_state_map, wn);
                _state->CopyFrom(s);
                if (_apply)
                    _client->Apply(wn, _state);
                break;

            case OPR_LABEL:
                s = (TState *)WN_MAP_Get(_state_map, wn);
                s->CopyFrom(_state);
                if (_apply)
                    _client->Apply(wn, _state);
                break;

            case OPR_GOTO:
            case OPR_REGION_EXIT:
                s = GetLabelState(wn);
                _state->CopyFrom(s);
                if (_apply)
                    _client->Apply(wn, _state);
                break;

            case OPR_TRUEBR:
            case OPR_FALSEBR:
                s = GetLabelState(wn);
                _state->Merge(s);
                s = (TState *)WN_MAP_Get(_state_map, wn);
                _change |= _state->Merge(s);
                s->CopyFrom(_state);
                TraverseBackwardExpr(wn);
                if (_apply) {
                    FmtAssert(!_change, ("DF state changed in APPLY mode!"));
                    _client->Apply(wn, _state);
                }
                break;

            case OPR_FUNC_ENTRY:
                _state->Empty();
                TraverseBackward(WN_func_body(wn));
                if (_apply)
                    _client->Apply(wn, _state);
                break;

            case OPR_IO:
            case OPR_GOTO_OUTER_BLOCK:
            case OPR_SWITCH:
            case OPR_CASEGOTO:
            case OPR_COMPGOTO:
            case OPR_XGOTO:
            case OPR_AGOTO:
            case OPR_ALTENTRY:
                FmtAssert(0, ("Unhandled control flow"));
                break;

            default:
                FmtAssert(!OPCODE_is_scf(WN_opcode(wn)), ("structured control flow not yet supported for backward dflow"));
                FmtAssert(OPCODE_is_stmt(WN_opcode(wn)), ("expected statement"));
                TraverseBackwardExpr(wn);
                break;
            } 
        }; // TraverseBackward
};

#endif

