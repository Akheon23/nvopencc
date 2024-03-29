/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

/* 
   Copyright 2003, 2004, 2005, 2006 PathScale, Inc.  All Rights Reserved.
   File modified October 3, 2003 by PathScale, Inc. to update Open64 C/C++ 
   front-ends to GNU 3.3.1 release.
 */

/* This file contains the definitions and documentation for the common
   tree codes used in the GNU C and C++ compilers (see c-common.def
   for the standard codes).  
   Copyright (C) 2000, 2001, 2002 Free Software Foundation, Inc.
   Written by Benjamin Chelf (chelf@codesourcery.com).

This file is part of GCC.

GCC is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2, or (at your option) any later
version.

GCC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with GCC; see the file COPYING.  If not, write to the Free
Software Foundation, 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.  */

#include "config.h"
#include "system.h"
#ifdef SGI_MONGOOSE
// To get typdef tree
#include "rtl.h"
#endif /* SGI_MONGOOSE */
#include "tree.h"
#include "function.h"
#include "splay-tree.h"
#include "varray.h"
#include "c-common.h"
#include "except.h"
#include "toplev.h"
#include "flags.h"
#include "ggc.h"
#ifndef SGI_MONGOOSE
#include "rtl.h"
#endif /* SGI_MONGOOSE */
#include "expr.h"
#include "output.h"
#include "timevar.h"
#include "predict.h"
#ifdef KEY
#include "wfe_dst.h"
#include "omp_types.h"
#include "omp_directive.h"
#endif // KEY
#if defined(TARG_SL) || defined(TARG_MIPS) || defined(TARG_NVISA)
#include "wfe_stmt.h"
#endif

#ifdef TARG_NVISA
#include "wfe_dst.h"
#endif

/* If non-NULL, the address of a language-specific function for
   expanding statements.  */
void (*lang_expand_stmt) PARAMS ((tree));

/* If non-NULL, the address of a language-specific function for
   expanding a DECL_STMT.  After the language-independent cases are
   handled, this function will be called.  If this function is not
   defined, it is assumed that declarations other than those for
   variables and labels do not require any RTL generation.  */
void (*lang_expand_decl_stmt) PARAMS ((tree));

/* Create an empty statement tree rooted at T.  */

void
begin_stmt_tree (t)
     tree *t;
{
  /* We create a trivial EXPR_STMT so that last_tree is never NULL in
     what follows.  We remove the extraneous statement in
     finish_stmt_tree.  */
  *t = build_nt (EXPR_STMT, void_zero_node);
  last_tree = *t;
  last_expr_type = NULL_TREE;
  last_expr_filename = input_filename;
}

/* T is a statement.  Add it to the statement-tree.  */

tree
add_stmt (t)
     tree t;
{
  if (input_filename != last_expr_filename)
    {
      /* If the filename has changed, also add in a FILE_STMT.  Do a string
	 compare first, though, as it might be an equivalent string.  */
      int add = (strcmp (input_filename, last_expr_filename) != 0);
      last_expr_filename = input_filename;
      if (add)
	{
	  tree pos = build_nt (FILE_STMT, get_identifier (input_filename));
	  add_stmt (pos);
	}
    }

  /* Add T to the statement-tree.  */
  TREE_CHAIN (last_tree) = t;
  last_tree = t;
  
  /* When we expand a statement-tree, we must know whether or not the
     statements are full-expressions.  We record that fact here.  */
  STMT_IS_FULL_EXPR_P (last_tree) = stmts_are_full_exprs_p ();

  /* Keep track of the number of statements in this function.  */
  if (current_function_decl)
    ++DECL_NUM_STMTS (current_function_decl);

  return t;
}

/* Create a declaration statement for the declaration given by the
   DECL.  */

void
add_decl_stmt (decl)
     tree decl;
{
  tree decl_stmt;

  /* We need the type to last until instantiation time.  */
  decl_stmt = build_stmt (DECL_STMT, decl);
  add_stmt (decl_stmt); 
}

/* Add a scope-statement to the statement-tree.  BEGIN_P indicates
   whether this statements opens or closes a scope.  PARTIAL_P is true
   for a partial scope, i.e, the scope that begins after a label when
   an object that needs a cleanup is created.  If BEGIN_P is nonzero,
   returns a new TREE_LIST representing the top of the SCOPE_STMT
   stack.  The TREE_PURPOSE is the new SCOPE_STMT.  If BEGIN_P is
   zero, returns a TREE_LIST whose TREE_VALUE is the new SCOPE_STMT,
   and whose TREE_PURPOSE is the matching SCOPE_STMT with
   SCOPE_BEGIN_P set.  */

tree
add_scope_stmt (begin_p, partial_p)
     int begin_p;
     int partial_p;
{
  tree *stack_ptr = current_scope_stmt_stack ();
  tree ss;
  tree top = *stack_ptr;

#ifdef TARG_NVISA
  if (parser_scope_level >= NESTED_BLOCK_START_LEVEL)
    seen_scope_stmt = true;
#endif

  /* Build the statement.  */
  ss = build_stmt (SCOPE_STMT, NULL_TREE);
  SCOPE_BEGIN_P (ss) = begin_p;
  SCOPE_PARTIAL_P (ss) = partial_p;

  /* Keep the scope stack up to date.  */
  if (begin_p)
    {
      top = tree_cons (ss, NULL_TREE, top);
      *stack_ptr = top;
#ifdef TARG_NVISA
      parser_scope_level++;
      WFE_Increment_Scope_Level_Count(parser_scope_level);
#endif
    }
  else
    {
      TREE_VALUE (top) = ss;
      *stack_ptr = TREE_CHAIN (top);
#ifdef TARG_NVISA
      parser_scope_level--;
#endif
    }

  /* Add the new statement to the statement-tree.  */
  add_stmt (ss);

  return top;
}

/* Finish the statement tree rooted at T.  */

void
finish_stmt_tree (t)
     tree *t;
{
  tree stmt;
  
  /* Remove the fake extra statement added in begin_stmt_tree.  */
  stmt = TREE_CHAIN (*t);
  *t = stmt;
  last_tree = NULL_TREE;

  if (cfun && stmt)
    {
      /* The line-number recorded in the outermost statement in a function
	 is the line number of the end of the function.  */
      STMT_LINENO (stmt) = lineno;
      STMT_LINENO_FOR_FN_P (stmt) = 1;
    }
}

/* Build a generic statement based on the given type of node and
   arguments. Similar to `build_nt', except that we set
   STMT_LINENO to be the current line number.  */
/* ??? This should be obsolete with the lineno_stmt productions
   in the grammar.  */

tree
build_stmt VPARAMS ((enum tree_code code, ...))
{
  tree t;
  int length;
  int i;

  VA_OPEN (p, code);
  VA_FIXEDARG (p, enum tree_code, code);

  t = make_node (code);
  length = TREE_CODE_LENGTH (code);
  STMT_LINENO (t) = lineno;

  for (i = 0; i < length; i++)
    TREE_OPERAND (t, i) = va_arg (p, tree);

  VA_CLOSE (p);
  return t;
}

/* Some statements, like for-statements or if-statements, require a
   condition.  This condition can be a declaration.  If T is such a
   declaration it is processed, and an expression appropriate to use
   as the condition is returned.  Otherwise, T itself is returned.  */

tree
expand_cond (t)
     tree t;
{
  if (t && TREE_CODE (t) == TREE_LIST)
    {
      expand_stmt (TREE_PURPOSE (t));
      return TREE_VALUE (t);
    }
  else 
    return t;
}

/* Create RTL for the local static variable DECL.  */

void
make_rtl_for_local_static (decl)
     tree decl;
{
  const char *asmspec = NULL;

  /* If we inlined this variable, we could see it's declaration
     again.  */
  if (TREE_ASM_WRITTEN (decl))
    return;

  /* If the DECL_ASSEMBLER_NAME is not the same as the DECL_NAME, then
     either we already created RTL for this DECL (and since it was a
     local variable, its DECL_ASSEMBLER_NAME got hacked up to prevent
     clashes with other local statics with the same name by a previous
     call to make_decl_rtl), or the user explicitly requested a
     particular assembly name for this variable, using the GNU
     extension for this purpose:

       int i asm ("j");

     There's no way to know which case we're in, here.  But, it turns
     out we're safe.  If there's already RTL, then
     rest_of_decl_compilation ignores the ASMSPEC parameter, so we
     may as well not pass it in.  If there isn't RTL, then we didn't
     already create RTL, which means that the modification to
     DECL_ASSEMBLER_NAME came only via the explicit extension.  */
  if (DECL_ASSEMBLER_NAME (decl) != DECL_NAME (decl)
      && !DECL_RTL_SET_P (decl))
    asmspec = IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl));

  rest_of_decl_compilation (decl, asmspec, /*top_level=*/0, /*at_end=*/0);
}

/* Let the back-end know about DECL.  */

void
emit_local_var (decl)
     tree decl;
{
  /* Create RTL for this variable.  */
  if (!DECL_RTL_SET_P (decl))
    {
      if (DECL_C_HARD_REGISTER (decl))
	/* The user specified an assembler name for this variable.
	   Set that up now.  */
	rest_of_decl_compilation
	  (decl, IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl)),
	   /*top_level=*/0, /*at_end=*/0);
      else
	expand_decl (decl);
    }

  /* Actually do the initialization.  */
  if (stmts_are_full_exprs_p ())
    expand_start_target_temps ();

  expand_decl_init (decl);

  if (stmts_are_full_exprs_p ())
    expand_end_target_temps ();
}

/* Helper for generating the RTL at the beginning of a scope.  */

void
genrtl_do_pushlevel ()
{
  emit_line_note (input_filename, lineno);
  clear_last_expr ();
}

/* Generate the RTL for DESTINATION, which is a GOTO_STMT.  */

void
genrtl_goto_stmt (destination)
     tree destination;
{
  if (TREE_CODE (destination) == IDENTIFIER_NODE)
    abort ();
  
  /* We warn about unused labels with -Wunused.  That means we have to
     mark the used labels as used.  */
  if (TREE_CODE (destination) == LABEL_DECL)
    TREE_USED (destination) = 1;
  
  emit_line_note (input_filename, lineno);
  
  if (TREE_CODE (destination) == LABEL_DECL)
    {
      label_rtx (destination);
      expand_goto (destination); 
    }
  else
    expand_computed_goto (destination);
}

/* Generate the RTL for EXPR, which is an EXPR_STMT.  Provided just
   for backward compatibility.  genrtl_expr_stmt_value() should be
   used for new code.  */

void
genrtl_expr_stmt (expr)
     tree expr;
{
  genrtl_expr_stmt_value (expr, -1, 1);
}

/* Generate the RTL for EXPR, which is an EXPR_STMT.  WANT_VALUE tells
   whether to (1) save the value of the expression, (0) discard it or
   (-1) use expr_stmts_for_value to tell.  The use of -1 is
   deprecated, and retained only for backward compatibility.
   MAYBE_LAST is nonzero if this EXPR_STMT might be the last statement
   in expression statement.  */

void 
genrtl_expr_stmt_value (expr, want_value, maybe_last)
     tree expr;
     int want_value, maybe_last;
{
  if (expr != NULL_TREE)
    {
      emit_line_note (input_filename, lineno);
      
      if (stmts_are_full_exprs_p ())
	expand_start_target_temps ();
      
      if (expr != error_mark_node)
	expand_expr_stmt_value (expr, want_value, maybe_last);
      
      if (stmts_are_full_exprs_p ())
	expand_end_target_temps ();
    }
}

/* Generate the RTL for T, which is a DECL_STMT.  */

void
genrtl_decl_stmt (t)
     tree t;
{
  tree decl;
  emit_line_note (input_filename, lineno);
  decl = DECL_STMT_DECL (t);
  /* If this is a declaration for an automatic local
     variable, initialize it.  Note that we might also see a
     declaration for a namespace-scope object (declared with
     `extern').  We don't have to handle the initialization
     of those objects here; they can only be declarations,
     rather than definitions.  */
  if (TREE_CODE (decl) == VAR_DECL 
      && !TREE_STATIC (decl)
      && !DECL_EXTERNAL (decl))
    {
      /* Let the back-end know about this variable.  */
      if (!anon_aggr_type_p (TREE_TYPE (decl)))
	emit_local_var (decl);
      else
	expand_anon_union_decl (decl, NULL_TREE, 
				DECL_ANON_UNION_ELEMS (decl));
    }
  else if (TREE_CODE (decl) == VAR_DECL && TREE_STATIC (decl))
    make_rtl_for_local_static (decl);
  else if (TREE_CODE (decl) == LABEL_DECL 
	   && C_DECLARED_LABEL_FLAG (decl))
    declare_nonlocal_label (decl);
  else if (lang_expand_decl_stmt)
    (*lang_expand_decl_stmt) (t);
}

/* Generate the RTL for T, which is an IF_STMT.  */

void
genrtl_if_stmt (t)
     tree t;
{
  tree cond;
  genrtl_do_pushlevel ();
  cond = expand_cond (IF_COND (t));
  emit_line_note (input_filename, lineno);
  expand_start_cond (cond, 0);
  if (THEN_CLAUSE (t))
    expand_stmt (THEN_CLAUSE (t));
#ifdef SGI_MONGOOSE
  if (THEN_CLAUSE (t))
    dump_parse_tree ("then clause", THEN_CLAUSE (t));
#endif /* SGI_MONGOOSE */
  if (ELSE_CLAUSE (t))
    {
      expand_start_else ();
      expand_stmt (ELSE_CLAUSE (t));
#ifdef SGI_MONGOOSE
      dump_parse_tree ("else clause", ELSE_CLAUSE (t));
#endif /* SGI_MONGOOSE */
    }
  expand_end_cond ();
}

/* Generate the RTL for T, which is a WHILE_STMT.  */

void
genrtl_while_stmt (t)
     tree t;
{
  tree cond;
  emit_nop ();
  emit_line_note (input_filename, lineno);
  expand_start_loop (1); 
  genrtl_do_pushlevel ();

  cond = expand_cond (WHILE_COND (t));
  emit_line_note (input_filename, lineno);
  expand_exit_loop_top_cond (0, cond);
  genrtl_do_pushlevel ();
  
  expand_stmt (WHILE_BODY (t));

#ifdef TARG_NVISA
  if (debug_info_level >= DINFO_LEVEL_NORMAL && optimize ==0)
     WFE_Insert_End_Braces_Label(lineno);
#endif

  expand_end_loop ();
}

/* Generate the RTL for T, which is a DO_STMT.  */

void
genrtl_do_stmt (t)
     tree t;
{
  tree cond = DO_COND (t);

  /* Recognize the common special-case of do { ... } while (0) and do
     not emit the loop widgetry in this case.  In particular this
     avoids cluttering the rtl with dummy loop notes, which can affect
     alignment of adjacent labels.  COND can be NULL due to parse
     errors.  */
#ifndef SGI_MONGOOSE
  // If we have a break inside a do{ .. break; .. } while (0) then
  // it will be turned to a nop. We would like to preserve this break because
  // it will control flow. An example is in 186.crafty/src/lookup.c:63
  if (!cond || integer_zerop (cond))
    {
      expand_start_null_loop ();
      expand_stmt (DO_BODY (t));
      expand_end_null_loop ();
    }
  else
#endif /* SGI_MONGOOSE */
    {
      emit_nop ();
      emit_line_note (input_filename, lineno);
      expand_start_loop_continue_elsewhere (1);

      expand_stmt (DO_BODY (t));

      expand_loop_continue_here ();
      cond = expand_cond (cond);
      emit_line_note (input_filename, lineno);
      expand_exit_loop_if_false (0, cond);
      expand_end_loop ();
    }
}

/* Build the node for a return statement and return it.  */

tree
build_return_stmt (expr)
     tree expr;
{
  return (build_stmt (RETURN_STMT, expr));
}

/* Generate the RTL for STMT, which is a RETURN_STMT.  */

void
genrtl_return_stmt (stmt)
     tree stmt;
{
  tree expr;

  expr = RETURN_STMT_EXPR (stmt);

  emit_line_note (input_filename, lineno);
  if (!expr)
    expand_null_return ();
  else
    {
      expand_start_target_temps ();
      expand_return (expr);
      expand_end_target_temps ();
    }
}

#ifdef KEY
extern struct nesting * get_loop_stack (void);
#endif // KEY

/* Generate the RTL for T, which is a FOR_STMT.  */

void
genrtl_for_stmt (t)
     tree t;
{
  tree cond;
  const char *saved_filename;
  int saved_lineno;

#ifdef KEY
  if (TREE_ADDRESSABLE (t))
  { // Generate a DO-loop for OpenMP
    tree init_expr = FOR_INIT_STMT (t);
    // Make GNU believe we have a loop here
    expand_start_null_loop ();
    if (TREE_CODE (init_expr) == EXPR_STMT)
      init_expr = TREE_OPERAND (init_expr, 0);
    if (TREE_CODE (init_expr) != MODIFY_EXPR && 
        TREE_CODE (init_expr) != VAR_DECL)
      abort();
    expand_start_do_loop (init_expr, FOR_COND (t), FOR_EXPR (t), get_loop_stack());
    expand_stmt (FOR_BODY (t));
    expand_end_do_loop (get_loop_stack());
    // Took care of GNU
    expand_end_null_loop ();
  }
  else
#endif // KEY
  { // the normal case
#ifdef TARG_NVISA
   int end_lineno;
#endif

  if (NEW_FOR_SCOPE_P (t))
    genrtl_do_pushlevel ();

  expand_stmt (FOR_INIT_STMT (t));

  /* Expand the initialization.  */
  emit_nop ();
  emit_line_note (input_filename, lineno);
  expand_start_loop_continue_elsewhere (1); 
  genrtl_do_pushlevel ();
  cond = expand_cond (FOR_COND (t));

  /* Save the filename and line number so that we expand the FOR_EXPR
     we can reset them back to the saved values.  */
  saved_filename = input_filename;
  saved_lineno = lineno;

  /* Expand the condition.  */
  emit_line_note (input_filename, lineno);
  if (cond)
    expand_exit_loop_top_cond (0, cond);

  /* Expand the body.  */
  genrtl_do_pushlevel ();
  expand_stmt (FOR_BODY (t));
#ifdef TARG_NVISA
  end_lineno = lineno;
#endif

  /* Expand the increment expression.  */
  input_filename = saved_filename;
  lineno = saved_lineno;
  emit_line_note (input_filename, lineno);
  expand_loop_continue_here ();
  if (FOR_EXPR (t))
    genrtl_expr_stmt (FOR_EXPR (t));

#ifdef TARG_NVISA
  if (debug_info_level >= DINFO_LEVEL_NORMAL && optimize == 0)
     WFE_Insert_End_Braces_Label(end_lineno);
#endif

  expand_end_loop ();
  }
}

/* Build a break statement node and return it.  */

tree
build_break_stmt ()
{
  return (build_stmt (BREAK_STMT));
}

/* Generate the RTL for a BREAK_STMT.  */

void
genrtl_break_stmt ()
{
  emit_line_note (input_filename, lineno);
  if ( ! expand_exit_something ())
    error ("break statement not within loop or switch");
}

/* Build a continue statement node and return it.  */

tree
build_continue_stmt ()
{
  return (build_stmt (CONTINUE_STMT));
}

#ifdef KEY

/* Build a omp-marker statement node and return it */
tree
build_omp_stmt (enum omp_tree_type c, void * p)
{
  tree t = build_stmt (OMP_MARKER_STMT);
  t->omp.choice = c;
  t->omp.omp_clause_list = p;

  return t;
}
#endif

/* Generate the RTL for a CONTINUE_STMT.  */

void
genrtl_continue_stmt ()
{
  emit_line_note (input_filename, lineno);
  if (! expand_continue_loop (0))
    error ("continue statement not within a loop");   
}

/* Generate the RTL for T, which is a SCOPE_STMT.  */

void
genrtl_scope_stmt (t)
     tree t;
{
  tree block = SCOPE_STMT_BLOCK (t);

  if (!SCOPE_NO_CLEANUPS_P (t))
    {
      if (SCOPE_BEGIN_P (t))
	expand_start_bindings_and_block (2 * SCOPE_NULLIFIED_P (t), block);
      else if (SCOPE_END_P (t))
	expand_end_bindings (NULL_TREE, !SCOPE_NULLIFIED_P (t), 0);

#ifdef TARG_NVISA
      if (SCOPE_BEGIN_P (t))
      {
         scope_dst_stack_index++;
         WFE_Increment_Scope_Level_Count(scope_dst_stack_index);

         if (debug_info_level >= 2 && optimize == 0)
            WFE_Create_Lexical_Block();

      }

      if (SCOPE_END_P (t)) 
      {

         if (debug_info_level >= 2 && optimize == 0)
            WFE_Finish_Lexical_Block();

         scope_dst_stack_index--;
      }
#endif
    }
  else if (!SCOPE_NULLIFIED_P (t))
    {
      rtx note = emit_note (NULL,
			    (SCOPE_BEGIN_P (t) 
			     ? NOTE_INSN_BLOCK_BEG
			     : NOTE_INSN_BLOCK_END));
      NOTE_BLOCK (note) = block;
    }

  /* If we're at the end of a scope that contains inlined nested
     functions, we have to decide whether or not to write them out.  */
  if (block && SCOPE_END_P (t))
    {
      tree fn;

      for (fn = BLOCK_VARS (block); fn; fn = TREE_CHAIN (fn))
	{
	  if (TREE_CODE (fn) == FUNCTION_DECL 
	      && DECL_CONTEXT (fn) == current_function_decl
	      && DECL_SAVED_INSNS (fn)
	      && !TREE_ASM_WRITTEN (fn)
	      && TREE_ADDRESSABLE (fn))
	    {
	      push_function_context ();
	      output_inline_function (fn);
	      pop_function_context ();
	    }
	}
    }
}

/* Generate the RTL for T, which is a SWITCH_STMT.  */

void
genrtl_switch_stmt (t)
     tree t;
{
  tree cond;
  genrtl_do_pushlevel ();
 
  cond = expand_cond (SWITCH_COND (t));
  if (cond == error_mark_node)
    /* The code is in error, but we don't want expand_end_case to
       crash.  */
    cond = boolean_false_node;

  emit_line_note (input_filename, lineno);
  expand_start_case (1, cond, TREE_TYPE (cond), "switch statement");
  expand_stmt (SWITCH_BODY (t));
  expand_end_case_type (cond, SWITCH_TYPE (t));
}

/* Create a CASE_LABEL tree node and return it.  */

tree
build_case_label (low_value, high_value, label_decl)
     tree low_value;
     tree high_value;
     tree label_decl;
{
  return build_stmt (CASE_LABEL, low_value, high_value, label_decl);
}


/* Generate the RTL for a CASE_LABEL.  */

void 
genrtl_case_label (case_label)
     tree case_label;
{
  tree duplicate;
  tree cleanup;

  cleanup = last_cleanup_this_contour ();
  if (cleanup)
    {
      static int explained = 0;
      warning ("destructor needed for `%#D'", (TREE_PURPOSE (cleanup)));
      warning ("where case label appears here");
      if (!explained)
	{
	  warning ("(enclose actions of previous case statements requiring destructors in their own scope.)");
	  explained = 1;
	}
    }

  add_case_node (CASE_LOW (case_label), CASE_HIGH (case_label), 
		 CASE_LABEL_DECL (case_label), &duplicate);
}

/* Generate the RTL for T, which is a COMPOUND_STMT.  */

void
genrtl_compound_stmt (t)
    tree t;
{
#ifdef ENABLE_CHECKING
  struct nesting *n = current_nesting_level ();
#endif

  expand_stmt (COMPOUND_BODY (t));

#ifdef ENABLE_CHECKING
  /* Make sure that we've pushed and popped the same number of levels.  */
  if (!COMPOUND_STMT_NO_SCOPE (t) && n != current_nesting_level ())
    abort ();
#endif
}

/* Generate the RTL for an ASM_STMT.  */

void
genrtl_asm_stmt (cv_qualifier, string, output_operands,
		 input_operands, clobbers, asm_input_p)
     tree cv_qualifier;
     tree string;
     tree output_operands;
     tree input_operands;
     tree clobbers;
     int asm_input_p;
{
  if (cv_qualifier != NULL_TREE
      && cv_qualifier != ridpointers[(int) RID_VOLATILE])
    {
      warning ("%s qualifier ignored on asm",
	       IDENTIFIER_POINTER (cv_qualifier));
      cv_qualifier = NULL_TREE;
    }

  emit_line_note (input_filename, lineno);
  if (asm_input_p)
    expand_asm (string, cv_qualifier != NULL_TREE);
  else
    c_expand_asm_operands (string, output_operands, input_operands, 
			   clobbers, cv_qualifier != NULL_TREE,
			   input_filename, lineno);
}

/* Generate the RTL for a DECL_CLEANUP.  */

void 
genrtl_decl_cleanup (t)
     tree t;
{
  tree decl = CLEANUP_DECL (t);
  if (!decl || (DECL_SIZE (decl) && TREE_TYPE (decl) != error_mark_node))
    expand_decl_cleanup_eh (decl, CLEANUP_EXPR (t), CLEANUP_EH_ONLY (t));
}

/* We're about to expand T, a statement.  Set up appropriate context
   for the substitution.  */

void
prep_stmt (t)
     tree t;
{
  if (!STMT_LINENO_FOR_FN_P (t))
    lineno = STMT_LINENO (t);
  current_stmt_tree ()->stmts_are_full_exprs_p = STMT_IS_FULL_EXPR_P (t);
}

#ifdef KEY
extern void WFE_Expand_Pragma (tree);
extern void
process_omp_stmt (tree t)
{
  switch (t->omp.choice)
  {
    case parallel_dir_b:
      expand_start_parallel ((struct parallel_clause_list *)t->omp.omp_clause_list);
      break;
    case parallel_dir_e:
      expand_end_parallel ();
      break;
    case for_dir_b:
      expand_start_for ((struct for_clause_list *)t->omp.omp_clause_list);
      break;
    case for_dir_e:
      expand_end_for ();
      break;
    case sections_cons_b:
      expand_start_sections ((struct sections_clause_list *)t->omp.omp_clause_list);
      break;
    case sections_cons_e:
      expand_end_sections ();
      break;
    case section_cons_b:
      expand_start_section ();
      break;
    case section_cons_e:
      expand_end_section ();
      break;
    case single_cons_b:
      expand_start_single ((struct single_clause_list *)t->omp.omp_clause_list);
      break;
    case single_cons_e:
      expand_end_single ();
      break;
    case par_for_cons_b:
      expand_start_parallel_for ((struct parallel_for_clause_list *)t->omp.omp_clause_list);
      break;
    case par_for_cons_e:
      expand_end_parallel_for ();
      break;
    case par_sctn_cons_b:
      expand_start_parallel_sections ((struct parallel_sections_clause_list *)t->omp.omp_clause_list);
      break;
    case par_sctn_cons_e:
      expand_end_parallel_sections ();
      break;
    case master_cons_b:
      expand_start_master ();
      break;
    case master_cons_e:
      expand_end_master ();
      break;
    case critical_cons_b:
      expand_start_critical ((tree)t->omp.omp_clause_list);
      break;
    case critical_cons_e:
      expand_end_critical ();
      break;
    case barrier_dir:
      expand_barrier ();
      break;
    case flush_dir:
      expand_flush ((tree)t->omp.omp_clause_list);
      break;
    case atomic_cons:
      check_atomic_expression ((tree)t->omp.omp_clause_list);
      expand_start_atomic ();
      expand_expr_stmt_value ((tree)t->omp.omp_clause_list, 0, 1);
      expand_end_atomic ();
      break;
    case ordered_cons_b:
      expand_start_ordered ();
      break;
    case ordered_cons_e:
      expand_end_ordered ();
      break;
    case options_dir:
    case exec_freq_dir:
    case unroll_dir:
      WFE_Expand_Pragma (t);
      break;
#ifdef TARG_SL  // fork_joint
    case sl2_sections_cons_b:
    case sl2_minor_sections_cons_b:		
      expand_start_sl2_sections (t->omp.choice == sl2_minor_sections_cons_b);
      break;
    case sl2_sections_cons_e:
      expand_end_sl2_sections ();
      break;
    case sl2_section_cons_b:
    case sl2_minor_section_cons_b:		
      expand_start_sl2_section (t->omp.choice == sl2_minor_section_cons_b);
      break;
    case sl2_section_cons_e:
    case sl2_minor_section_cons_e:		
      expand_end_sl2_section ();
      break;
#endif 

    default:
      abort();
  }
}
#endif // KEY

/* Generate the RTL for the statement T, its substatements, and any
   other statements at its nesting level.  */

void
expand_stmt (t)
     tree t;
{
  while (t && t != error_mark_node)
    {
      int saved_stmts_are_full_exprs_p;

      /* Set up context appropriately for handling this statement.  */
      saved_stmts_are_full_exprs_p = stmts_are_full_exprs_p ();
      prep_stmt (t);

      switch (TREE_CODE (t))
	{
	case FILE_STMT:
	  input_filename = FILE_STMT_FILENAME (t);
#ifdef KEY // Bug 10632, 8895: File changed. Don't worry linenum --not used !!
          WFE_Set_Line_And_File (lineno, input_filename);
#endif
	  break;

	case RETURN_STMT:
	  genrtl_return_stmt (t);
	  break;

	case EXPR_STMT:
	  genrtl_expr_stmt_value (EXPR_STMT_EXPR (t), TREE_ADDRESSABLE (t),
				  TREE_CHAIN (t) == NULL
				  || (TREE_CODE (TREE_CHAIN (t)) == SCOPE_STMT
				      && TREE_CHAIN (TREE_CHAIN (t)) == NULL));
	  break;

	case DECL_STMT:
	  genrtl_decl_stmt (t);
	  break;

	case FOR_STMT:
	  genrtl_for_stmt (t);
	  break;

	case WHILE_STMT:
	  genrtl_while_stmt (t);
	  break;

	case DO_STMT:
	  genrtl_do_stmt (t);
	  break;

	case IF_STMT:
	  genrtl_if_stmt (t);
	  break;

	case COMPOUND_STMT:
	  genrtl_compound_stmt (t);
	  break;

	case BREAK_STMT:
	  genrtl_break_stmt ();
	  break;

	case CONTINUE_STMT:
	  genrtl_continue_stmt ();
	  break;

	case SWITCH_STMT:
	  genrtl_switch_stmt (t);
	  break;

	case CASE_LABEL:
	  genrtl_case_label (t);
	  break;

	case LABEL_STMT:
	  expand_label (LABEL_STMT_LABEL (t));
	  break;

	case GOTO_STMT:
	  /* Emit information for branch prediction.  */
	  if (!GOTO_FAKE_P (t)
	      && TREE_CODE (GOTO_DESTINATION (t)) == LABEL_DECL
	      && flag_guess_branch_prob)
	    {
	      rtx note = emit_note (NULL, NOTE_INSN_PREDICTION);

	      NOTE_PREDICTION (note) = NOTE_PREDICT (PRED_GOTO, NOT_TAKEN);
	    }
	  genrtl_goto_stmt (GOTO_DESTINATION (t));
	  break;

	case ASM_STMT:
	  genrtl_asm_stmt (ASM_CV_QUAL (t), ASM_STRING (t),
			   ASM_OUTPUTS (t), ASM_INPUTS (t),
			   ASM_CLOBBERS (t), ASM_INPUT_P (t));
	  break;

	case SCOPE_STMT:
	  genrtl_scope_stmt (t);
	  break;

	case CLEANUP_STMT:
	  genrtl_decl_cleanup (t);
	  break;

#ifdef KEY
	case OMP_MARKER_STMT:
	  process_omp_stmt (t);
	  break;
#endif // KEY

#if defined(TARG_SL) || defined(TARG_MIPS)
	case FREQ_HINT_STMT:
	  WFE_Expand_Freq_Hint (t);
	  break;
#endif

	default:
	  if (lang_expand_stmt)
	    (*lang_expand_stmt) (t);
	  else 
	    abort ();
	  break;
	}

      /* Restore saved state.  */
      current_stmt_tree ()->stmts_are_full_exprs_p
	= saved_stmts_are_full_exprs_p;

      /* Go on to the next statement in this scope.  */
      t = TREE_CHAIN (t);
    }
}
