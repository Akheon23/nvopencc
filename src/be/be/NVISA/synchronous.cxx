/*
 * Copyright 2009-2010 NVIDIA Corporation.  All rights reserved.
 */

//
// Proposed synchronous construct prototype. The user code is expected 
// to containg synchronous blocks specified as follows:
// __synchronous_start(W);
// S1;
// S2;
// S3;
// __synchronous_end();
//
//
// This prototype does not support use of synchronous  inside divergent
// region of code. The user code will get transformed to:
//
// __synclevel(W)
// S1;
// __synclevel(W)
// S2;
// __synclevel(W)
// S3;
// __synclevel(W)
//
// each __synclevel(W) transforms to:
// FORWARD_BARRIER;
// if (W > machine_warp_width || (machine_warp_width % W != 0))
//      __syncthreads();
// BACKWARD_BARRIER;
//
// All non-local memory accesses by statements inside the synchronous region are marked as volatile.
//
//
// The entry function for the code is Process_Synchronous().
#include "defs.h"
#include "symtab.h"
#include "wn.h"

#include "wn_util.h"
#include "wn_tree_util.h"
#include "wn_lower.h"
#include "const.h"

#include "synchronous.h"


// ====================================================================
//
// Description: Insert a whirl node before a statement node
//              
// ====================================================================
static void Insert_Wn_Before_Stmt(WN *new_wn, WN *stmt)
{
    WN *prev, *next;

    // Note: assumes never called for first statement in a block!
    prev = WN_prev(stmt);
    FmtAssert(prev, ("unhandled: insert before first stmt in block"));
    WN_next(prev) = new_wn;
    WN_prev(new_wn) = prev;

    WN_prev(stmt) = new_wn;
    WN_next(new_wn) = stmt;
} // Insert_Wn_Before_Stmt


// ====================================================================
//
// Description: Create whirl node to load value of warp size
//              
// ====================================================================
static WN *Load_Warp_Size(ST *st_warpsize)
{
   WN *wn;
    // TODO: OCG knows the value of WARPSIZE, but the tesla backend
    // does not yet constant fold expressions all instructions
    // So, for the evaluation prototype, hardcode the warpsize value to "32"
    // (otherwise, generated binary code will much slower - skewing performance 
    // impact measurements)
   
   // FINAL VERSION:
    //   wn = WN_Ldid(MTYPE_I4, 0, st_warpsize, MTYPE_To_TY(MTYPE_I4));
    // TEMPORARY HACK:
    wn = WN_Intconst(MTYPE_I4, 32);
    return wn;
} // Load_warp_size

// ====================================================================
//
// Description: Insert a barrier before a statement
//              
// ====================================================================
static void Insert_Barrier_Before(WN *stmt, ST *st_warpsize, int syncwidth)
{
    WN *wn, *wn2, *first, *second, *wn_test, *wn_if, *wn_body;
    
    // want to generate:
    //  FORWARD_BARRIER;
    //  if (syncwidth > WARPSIZE || (WARPSIZE % syncwidth  != 0)) {
    //      __syncthreads();
    //  }
    //  BACKWARD_BARRIER;

    // FORWARD BARRIER
    wn = WN_CreateBarrier(TRUE, 0);
    Insert_Wn_Before_Stmt(wn, stmt);

    // if block
    wn = WN_Intconst(MTYPE_I4, syncwidth);
    wn2 = Load_Warp_Size(st_warpsize);
    first = WN_GT(MTYPE_I4, wn, wn2);

    wn = Load_Warp_Size(st_warpsize);
    wn2 = WN_Intconst(MTYPE_I4, syncwidth);
    wn = WN_Binary(OPR_REM, MTYPE_I4, wn, wn2);
    wn2 = WN_Intconst(MTYPE_I4, 0);
    second = WN_NE(MTYPE_I4, wn, wn2);

    wn_test = WN_Binary(OPR_BIOR, MTYPE_I4, first, second);
    wn_body = WN_CreateBlock();
    wn = WN_Create_Intrinsic(OPR_INTRINSIC_CALL, MTYPE_V, MTYPE_V, INTRN_SYNCTHREADS, 0, NULL);
    WN_INSERT_BlockAfter(wn_body, WN_last(wn_body), wn);

    wn_if = WN_CreateIf(wn_test, wn_body, WN_CreateBlock());
    Insert_Wn_Before_Stmt(wn_if, stmt);

    // BACKWARD_BARRIER
    wn = WN_CreateBarrier(FALSE, 0);
    Insert_Wn_Before_Stmt(wn, stmt);

} // Insert_Barrier_Before

// ====================================================================
//
// Description: Insert barriers before each statement in the synchronous
//              block
// ====================================================================
static void Insert_Barriers(WN *start, WN *end, int syncwidth)
{
    WN_ITER *itr;
    WN *stmt, *wn, *next;
    ST *st_warpsize;
    INT i;
    OPERATOR opr;

    // find symbol refering to warpsize.
    st_warpsize = NULL;
    FOREACH_SYMBOL(GLOBAL_SYMTAB, st_warpsize, i) {
        if (strcmp(ST_name(st_warpsize), "warpSize") == 0) {
            break;
        }
    }
    FmtAssert(st_warpsize, ("unable to locate predefined symbol warpSize"));

    stmt = WN_next(start);
    while (stmt != end) {
        next = WN_next(stmt);
        Insert_Barrier_Before(stmt, st_warpsize, syncwidth);
        stmt = next;
    }
    Insert_Barrier_Before(end, st_warpsize, syncwidth);
} // Insert_barriers

// ====================================================================
//
// Description: Mark all loads and stores in the synchronous region as
//              volatile
// ====================================================================
static void Set_Memops_To_Volatile(WN *start, WN *end)
{
    WN *stmt, *wn, *next;
    WN_ITER *itr;
    ST *st;
    OPERATOR opr;
    TY_IDX ty;
    bool islocal;
    
    stmt = WN_next(start);
    while (stmt != end) {
        next = WN_next(stmt);
        for (itr = WN_WALK_TreeIter(stmt); itr != NULL; itr = WN_WALK_TreeNext(itr)) {
            wn = WN_ITER_wn(itr);
            opr = WN_operator(wn);
            if (OPERATOR_is_load(opr) || OPERATOR_is_store(opr)) {
                switch (opr) {
                
                case OPR_ILOAD:
                    ty = WN_load_addr_ty(wn);
                    FmtAssert(TY_kind(ty) == KIND_POINTER, ("expected pointer!"));
                    ty = TY_pointed(ty);
                    Set_TY_is_volatile(ty);
                    ty = Make_Pointer_Type(ty);
                    WN_set_load_addr_ty(wn, ty);
                    
                    ty = WN_ty(wn);
                    Set_TY_is_volatile(ty);
                    WN_set_ty(wn, ty);
                    break;
                
                case OPR_LDID:
                case OPR_STID:
                    st = &St_Table[WN_st_idx(wn)];
                    // only mark global and shared memory accesses as volatile
                    islocal = (ST_class(st) == CLASS_VAR && ST_sclass(st) == SCLASS_AUTO &&
                              !(ST_in_global_mem(st) || ST_in_shared_mem(st)));
                    if (!islocal)
                    {
                        ty = WN_ty(wn);
                        Set_TY_is_volatile(ty);
                        WN_set_ty(wn, ty);
                    }
                    break;

                case OPR_ISTORE:
                    ty = WN_ty(wn);
                    FmtAssert(TY_kind(ty) == KIND_POINTER, ("expected pointer!"));
                    ty = TY_pointed(ty);
                    Set_TY_is_volatile(ty);
                    ty = Make_Pointer_Type(ty);
                    WN_set_ty(wn, ty);
                    break;

                default:
                    FmtAssert(0, ("Hit unhandled mem load/store: %s inside synchronous", 
                                  OPERATOR_name(opr)));
                    break;
                }
            }
        }
        stmt = next;
    }
} // Set_Memops_To_Volatile

#if 0
// UPDATE: no longer required
// ====================================================================
//
// Description: For a store statement, compute the value to be stored 
//              into a temporary, and store from the temporary in a
//              separate statement.
// ====================================================================
static void Fix_Stores(WN *start, WN *end)
{
    WN *stmt, *next, *wn;
    WN_ITER *itr;
    OPERATOR opr;
    TY_IDX new_ty;
    PREG_NUM preg;
    ST *st_preg;
    TYPE_ID mtype;
    
    stmt = WN_next(start);
    while (stmt != end) {
        next = WN_next(stmt);
        
        opr = WN_operator(stmt);
        if (opr == OPR_STID) {
            mtype = WN_desc(stmt);
            preg = Create_Preg(mtype, "temp_val_expr");
            st_preg = MTYPE_To_PREG(mtype);
            wn = WN_Stid(mtype, preg, st_preg, MTYPE_To_TY(mtype), WN_kid0(stmt));
            insert_wn_before_stmt(wn, stmt);
            // barriers will be inserted after each statement in a later pass
            wn = WN_Ldid(mtype, preg, st_preg, MTYPE_To_TY(mtype));
            WN_kid0(stmt) = wn;
        } else if (opr == OPR_ISTORE) {
            mtype = WN_desc(stmt);
            preg = Create_Preg(mtype, "temp_val_expr");
            st_preg = MTYPE_To_PREG(mtype);
            wn = WN_Stid(mtype, preg, st_preg, MTYPE_To_TY(mtype), WN_kid0(stmt));
            insert_wn_before_stmt(wn, stmt);
            // barriers will be inserted after each statement in a later pass
            wn = WN_Ldid(mtype, preg, st_preg, MTYPE_To_TY(mtype));
            WN_kid0(stmt) = wn;

            // now to do the same for the address expression
            mtype = Pointer_Mtype2;
            preg = Create_Preg(mtype, "temp_addr_expr");
            st_preg = MTYPE_To_PREG(mtype);
            wn = WN_Stid(mtype, preg, st_preg, MTYPE_To_TY(mtype), WN_kid1(stmt));
            insert_wn_before_stmt(wn, stmt);
            // barriers will be inserted after each statement in a later pass
            wn = WN_Ldid(mtype, preg, st_preg, MTYPE_To_TY(mtype));
            WN_kid1(stmt) = wn;
        } else {
            FmtAssert(!OPERATOR_is_store(opr), ("unhandled store type!"));
        }

        stmt = next;
    }
} // Fix_Stores
#endif

// ====================================================================
//
// Description: Delete the __synchronous_start() and __synchronous_end()
//              calls.
// ====================================================================
static void Delete_Synchronous_Directive(WN *wn, WN *block)
{
    WN *kid, *next_wn;
    INT kidno;
    OPERATOR opr;
    
    opr = WN_operator(wn);
    switch(opr) {
    case OPR_BLOCK:
        kid = WN_first(wn);
        while (kid != NULL) {
            next_wn = WN_next(kid); // save it since kid might get deleted if it is STID
            Delete_Synchronous_Directive(kid, wn);
            kid = next_wn;
        }
        break;
    case OPR_INTRINSIC_CALL:
         if (WN_intrinsic(wn) == INTRN_SYNCHRONOUS_START ||
             WN_intrinsic(wn) == INTRN_SYNCHRONOUS_END) 
         {
            WN_EXTRACT_FromBlock(block, wn);
            WN_Delete(wn);
         }
        break;
    default:
        for (kidno = 0; kidno < WN_kid_count(wn); kidno++)
            Delete_Synchronous_Directive(WN_kid(wn, kidno), block);
        break;
    }
} // Delete_Synchronous_Directive

// ====================================================================
//
// Description: Main function. Process a synchronous block. A synchronous
//              is the set of statements between __synchronous_start()
//              and __synchronous_end()
// ====================================================================
void Process_Synchronous(WN *pu)
{
    WN *start, *end;
    WN_ITER *itr;
    WN *wn;
    OPERATOR opr;
    int syncwidth;
    
    // 1. look for synchronous_start...synchronous_end;
    for (itr = WN_WALK_TreeIter(pu); itr != NULL; itr = WN_WALK_TreeNext(itr)) {
        wn = WN_ITER_wn(itr);
        opr = WN_operator(wn);
        if (opr == OPR_INTRINSIC_CALL && WN_intrinsic(wn) == INTRN_SYNCHRONOUS_START) {
            start = wn;
            end = NULL;
            for (wn = WN_next(start); wn != NULL; wn = WN_next(wn)) {
                opr = WN_operator(wn);
                if (opr == OPR_INTRINSIC_CALL && WN_intrinsic(wn) == INTRN_SYNCHRONOUS_END) {
                    end = wn;
                    break;
                }
            }
            FmtAssert(end, ("Error: found synchronous_start but not synchronous_end!"));

            // 2. extract the synchronous parameter
            wn = WN_kid0(start);
            FmtAssert(WN_operator(wn) == OPR_PARM, ("expected parm node"));
            wn = WN_kid0(wn);
            FmtAssert(WN_operator(wn) == OPR_INTCONST, ("expected int const"));
            syncwidth = WN_const_val(wn);
            FmtAssert(syncwidth > 0, ("syncwidth needs to be > 0"));

            // 4. change all loads/stores to volatile
            Set_Memops_To_Volatile(start, end);

#if 0            
            // UPDATE: this is no longer required

            // 5. For all STID/ISTOREs, first write both address & value to temporaries
            // then use the temporaries to store to memory. This is because the address
            // or value expression may contain loads. If syncwidth is greater than warp
            // width, then to ensure synchronous semantics, we need to synchronize AFTER
            // the address and value expressions have been computed and BEFORE the store
            // occurs.
            // Fix_Stores(start, end);
#endif

            // 5. insert code motion barriers & conditionally, syncthreads
            Insert_Barriers(start, end, syncwidth);

        } // found synchronous_start

    } // iterating over entire program

    // 6. delete synchronous directives
    Delete_Synchronous_Directive(pu, NULL);

} // Process_Synchronous
