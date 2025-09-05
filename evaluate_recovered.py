# uncompyle6 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
# Embedded file name: /home/dell/DATA/mg/gmflow/evaluate.py
# Compiled at: 2025-08-19 15:31:37
# Size of source mod 2**32: 28189 bytes

-- Stacks of completed symbols:
START ::= |- stmts . 
_come_froms ::= \e__come_froms . COME_FROM
_come_froms ::= \e__come_froms . COME_FROM_LOOP
_come_froms ::= \e__come_froms COME_FROM . 
_come_froms ::= _come_froms . COME_FROM
_come_froms ::= _come_froms . COME_FROM_LOOP
_come_froms ::= _come_froms COME_FROM . 
_ifstmts_jump ::= \e_c_stmts_opt . COME_FROM
_ifstmts_jump ::= \e_c_stmts_opt . ELSE
_ifstmts_jump ::= \e_c_stmts_opt . JUMP_ABSOLUTE JUMP_FORWARD \e__come_froms
_ifstmts_jump ::= \e_c_stmts_opt . JUMP_ABSOLUTE JUMP_FORWARD _come_froms
_ifstmts_jump ::= \e_c_stmts_opt . come_froms
_ifstmts_jump ::= c_stmts_opt . COME_FROM
_ifstmts_jump ::= c_stmts_opt . ELSE
_ifstmts_jump ::= c_stmts_opt . JUMP_ABSOLUTE JUMP_FORWARD \e__come_froms
_ifstmts_jump ::= c_stmts_opt . JUMP_ABSOLUTE JUMP_FORWARD _come_froms
_ifstmts_jump ::= c_stmts_opt . come_froms
_ifstmts_jump ::= c_stmts_opt COME_FROM . 
_ifstmts_jump ::= c_stmts_opt come_froms . 
_ifstmts_jumpl ::= _ifstmts_jump . 
_ifstmts_jumpl ::= c_stmts . JUMP_BACK
_stmts ::= _stmts . stmt
_stmts ::= _stmts stmt . 
_stmts ::= stmt . 
and ::= expr . JUMP_IF_FALSE_OR_POP expr \e_come_from_opt
and ::= expr . JUMP_IF_FALSE_OR_POP expr come_from_opt
and ::= expr . jifop_come_from expr
and ::= expr . jmp_false expr
and ::= expr . jmp_false expr COME_FROM
and ::= expr . jmp_false expr jmp_false
and ::= expr jmp_false . expr
and ::= expr jmp_false . expr COME_FROM
and ::= expr jmp_false . expr jmp_false
and ::= expr jmp_false expr . 
and ::= expr jmp_false expr . COME_FROM
and ::= expr jmp_false expr . jmp_false
and ::= expr jmp_false expr jmp_false . 
and_not ::= expr . jmp_false expr POP_JUMP_IF_TRUE
and_not ::= expr jmp_false . expr POP_JUMP_IF_TRUE
and_not ::= expr jmp_false expr . POP_JUMP_IF_TRUE
assert2 ::= expr . jmp_true LOAD_GLOBAL expr CALL_FUNCTION_1 RAISE_VARARGS_1
assign ::= expr . DUP_TOP designList
assign ::= expr . store
assign ::= expr store . 
assign2 ::= expr . expr ROT_TWO store store
assign2 ::= expr expr . ROT_TWO store store
assign3 ::= expr . expr expr ROT_THREE ROT_TWO store store store
assign3 ::= expr expr . expr ROT_THREE ROT_TWO store store store
assign3 ::= expr expr expr . ROT_THREE ROT_TWO store store store
async_for_stmt38 ::= expr . async_for store for_block COME_FROM_FINALLY END_ASYNC_FOR
async_forelse_stmt38 ::= expr . GET_AITER SETUP_FINALLY GET_ANEXT LOAD_CONST YIELD_FROM POP_BLOCK store for_block COME_FROM_FINALLY END_ASYNC_FOR else_suite
attribute ::= expr . LOAD_ATTR
attribute ::= expr LOAD_ATTR . 
attribute37 ::= expr . LOAD_METHOD
attribute37 ::= expr LOAD_METHOD . 
aug_assign1 ::= expr . expr inplace_op ROT_THREE STORE_SUBSCR
aug_assign1 ::= expr . expr inplace_op store
aug_assign1 ::= expr expr . inplace_op ROT_THREE STORE_SUBSCR
aug_assign1 ::= expr expr . inplace_op store
aug_assign1 ::= expr expr inplace_op . ROT_THREE STORE_SUBSCR
aug_assign1 ::= expr expr inplace_op . store
aug_assign1 ::= expr expr inplace_op store . 
aug_assign2 ::= expr . DUP_TOP LOAD_ATTR expr inplace_op ROT_TWO STORE_ATTR
await_expr ::= expr . GET_AWAITABLE LOAD_CONST YIELD_FROM
bin_op ::= expr . expr binary_operator
bin_op ::= expr expr . binary_operator
bin_op ::= expr expr binary_operator . 
binary_operator ::= BINARY_ADD . 
binary_operator ::= BINARY_MODULO . 
binary_operator ::= BINARY_SUBTRACT . 
break ::= POP_TOP . BREAK_LOOP
c_stmts ::= _stmts . 
c_stmts ::= _stmts . lastc_stmt
c_stmts_opt ::= c_stmts . 
call ::= expr . CALL_METHOD_0
call ::= expr . pos_arg CALL_FUNCTION_1
call ::= expr . pos_arg CALL_METHOD_1
call ::= expr . pos_arg pos_arg CALL_FUNCTION_2
call ::= expr . pos_arg pos_arg CALL_METHOD_2
call ::= expr . pos_arg pos_arg pos_arg CALL_METHOD_3
call ::= expr CALL_METHOD_0 . 
call ::= expr pos_arg . CALL_FUNCTION_1
call ::= expr pos_arg . CALL_METHOD_1
call ::= expr pos_arg . pos_arg CALL_FUNCTION_2
call ::= expr pos_arg . pos_arg CALL_METHOD_2
call ::= expr pos_arg . pos_arg pos_arg CALL_METHOD_3
call ::= expr pos_arg CALL_FUNCTION_1 . 
call ::= expr pos_arg CALL_METHOD_1 . 
call ::= expr pos_arg pos_arg . CALL_FUNCTION_2
call ::= expr pos_arg pos_arg . CALL_METHOD_2
call ::= expr pos_arg pos_arg . pos_arg CALL_METHOD_3
call ::= expr pos_arg pos_arg CALL_METHOD_2 . 
call ::= expr pos_arg pos_arg pos_arg . CALL_METHOD_3
call_kw36 ::= expr . expr LOAD_CONST CALL_FUNCTION_KW_1
call_kw36 ::= expr . expr expr LOAD_CONST CALL_FUNCTION_KW_2
call_kw36 ::= expr . expr expr expr LOAD_CONST CALL_FUNCTION_KW_3
call_kw36 ::= expr . expr expr expr expr expr LOAD_CONST CALL_FUNCTION_KW_5
call_kw36 ::= expr expr . LOAD_CONST CALL_FUNCTION_KW_1
call_kw36 ::= expr expr . expr LOAD_CONST CALL_FUNCTION_KW_2
call_kw36 ::= expr expr . expr expr LOAD_CONST CALL_FUNCTION_KW_3
call_kw36 ::= expr expr . expr expr expr expr LOAD_CONST CALL_FUNCTION_KW_5
call_kw36 ::= expr expr LOAD_CONST . CALL_FUNCTION_KW_1
call_kw36 ::= expr expr expr . LOAD_CONST CALL_FUNCTION_KW_2
call_kw36 ::= expr expr expr . expr LOAD_CONST CALL_FUNCTION_KW_3
call_kw36 ::= expr expr expr . expr expr expr LOAD_CONST CALL_FUNCTION_KW_5
call_kw36 ::= expr expr expr LOAD_CONST . CALL_FUNCTION_KW_2
call_kw36 ::= expr expr expr LOAD_CONST CALL_FUNCTION_KW_2 . 
call_kw36 ::= expr expr expr expr . LOAD_CONST CALL_FUNCTION_KW_3
call_kw36 ::= expr expr expr expr . expr expr LOAD_CONST CALL_FUNCTION_KW_5
call_kw36 ::= expr expr expr expr LOAD_CONST . CALL_FUNCTION_KW_3
call_kw36 ::= expr expr expr expr LOAD_CONST CALL_FUNCTION_KW_3 . 
call_kw36 ::= expr expr expr expr expr . expr LOAD_CONST CALL_FUNCTION_KW_5
call_kw36 ::= expr expr expr expr expr expr . LOAD_CONST CALL_FUNCTION_KW_5
call_kw36 ::= expr expr expr expr expr expr LOAD_CONST . CALL_FUNCTION_KW_5
call_kw36 ::= expr expr expr expr expr expr LOAD_CONST CALL_FUNCTION_KW_5 . 
call_stmt ::= call . 
call_stmt ::= expr . POP_TOP
call_stmt ::= expr POP_TOP . 
cf_jf_else ::= come_froms . JUMP_FORWARD ELSE
cf_jump_back ::= COME_FROM . JUMP_BACK
cf_pt ::= COME_FROM . POP_TOP
classdefdeco1 ::= expr . classdefdeco1 CALL_FUNCTION_1
classdefdeco1 ::= expr . classdefdeco2 CALL_FUNCTION_1
come_from_loops ::= \e_come_from_loops . COME_FROM_LOOP
come_from_opt ::= COME_FROM . 
come_froms ::= COME_FROM . 
come_froms ::= come_froms . COME_FROM
come_froms ::= come_froms COME_FROM . 
compare ::= compare_single . 
compare_chained ::= expr . compared_chained_middle ROT_TWO POP_TOP \e__come_froms
compare_chained ::= expr . compared_chained_middle ROT_TWO POP_TOP _come_froms
compare_chained37 ::= expr . compared_chained_middlea_37
compare_chained37 ::= expr . compared_chained_middlec_37
compare_chained37_false ::= expr . compare_chained_right_false_37
compare_chained37_false ::= expr . compared_chained_middle_false_37
compare_chained37_false ::= expr . compared_chained_middleb_false_37
compare_chained_right_false_37 ::= expr . DUP_TOP ROT_THREE COMPARE_OP POP_JUMP_IF_FALSE compare_chained_righta_false_37 POP_TOP JUMP_BACK COME_FROM
compare_single ::= expr . expr COMPARE_OP
compare_single ::= expr expr . COMPARE_OP
compare_single ::= expr expr COMPARE_OP . 
compared_chained_middle ::= expr . DUP_TOP ROT_THREE COMPARE_OP JUMP_IF_FALSE_OR_POP compare_chained_right COME_FROM
compared_chained_middle ::= expr . DUP_TOP ROT_THREE COMPARE_OP JUMP_IF_FALSE_OR_POP compared_chained_middle COME_FROM
compared_chained_middle_false_37 ::= expr . DUP_TOP ROT_THREE COMPARE_OP POP_JUMP_IF_FALSE compare_chained_rightb_false_37 POP_TOP _jump COME_FROM
compared_chained_middle_false_37 ::= expr . DUP_TOP ROT_THREE COMPARE_OP POP_JUMP_IF_FALSE compare_chained_rightc_37 POP_TOP JUMP_FORWARD COME_FROM
compared_chained_middlea_37 ::= expr . DUP_TOP ROT_THREE COMPARE_OP POP_JUMP_IF_FALSE
compared_chained_middlea_37 ::= expr . DUP_TOP ROT_THREE COMPARE_OP POP_JUMP_IF_FALSE compare_chained_righta_37 COME_FROM POP_TOP COME_FROM
compared_chained_middleb_false_37 ::= expr . DUP_TOP ROT_THREE COMPARE_OP POP_JUMP_IF_FALSE compare_chained_rightb_false_37 POP_TOP _jump COME_FROM
compared_chained_middlec_37 ::= expr . DUP_TOP ROT_THREE COMPARE_OP POP_JUMP_IF_FALSE compare_chained_righta_37 POP_TOP
continues ::= _stmts . lastl_stmt continue
continues ::= _stmts lastl_stmt . continue
continues ::= lastl_stmt . continue
dict ::= kvlist_0 . 
dict_comp_func ::= BUILD_MAP_0 . LOAD_ARG for_iter store comp_iter JUMP_BACK ending_return
else_suite ::= suite_stmts . 
else_suite_opt ::= else_suite . 
else_suitec ::= c_stmts . 
expr ::= LOAD_CONST . 
expr ::= LOAD_FAST . 
expr ::= LOAD_GLOBAL . 
expr ::= LOAD_STR . 
expr ::= and . 
expr ::= attribute . 
expr ::= attribute37 . 
expr ::= bin_op . 
expr ::= call . 
expr ::= call_kw36 . 
expr ::= compare . 
expr ::= dict . 
expr ::= get_iter . 
expr ::= list . 
expr ::= subscript . 
expr_jit ::= expr . JUMP_IF_TRUE
expr_jitop ::= expr . JUMP_IF_TRUE_OR_POP
expr_jt ::= expr . jmp_true
expr_pjit ::= expr . POP_JUMP_IF_TRUE
expr_pjit_come_from ::= expr . POP_JUMP_IF_TRUE COME_FROM
expr_stmt ::= expr . POP_TOP
expr_stmt ::= expr POP_TOP . 
for38 ::= expr . get_for_iter store for_block
for38 ::= expr . get_for_iter store for_block JUMP_BACK
for38 ::= expr . get_for_iter store for_block JUMP_BACK POP_BLOCK
for38 ::= expr . get_iter store for_block JUMP_BACK
for38 ::= expr get_for_iter . store for_block
for38 ::= expr get_for_iter . store for_block JUMP_BACK
for38 ::= expr get_for_iter . store for_block JUMP_BACK POP_BLOCK
for38 ::= expr get_for_iter store . for_block
for38 ::= expr get_for_iter store . for_block JUMP_BACK
for38 ::= expr get_for_iter store . for_block JUMP_BACK POP_BLOCK
for38 ::= expr get_for_iter store for_block . 
for38 ::= expr get_for_iter store for_block . JUMP_BACK
for38 ::= expr get_for_iter store for_block . JUMP_BACK POP_BLOCK
for_block ::= \e__come_froms . l_stmts_opt _come_from_loops JUMP_BACK
for_block ::= \e__come_froms \e_l_stmts_opt . _come_from_loops JUMP_BACK
for_block ::= \e__come_froms l_stmts_opt . _come_from_loops JUMP_BACK
for_block ::= \e_l_stmts_opt . _come_froms JUMP_BACK
for_block ::= \e_l_stmts_opt . come_from_loops JUMP_BACK
for_block ::= \e_l_stmts_opt \e__come_froms . JUMP_BACK
for_block ::= \e_l_stmts_opt \e_come_from_loops . JUMP_BACK
for_block ::= l_stmts . 
for_block ::= l_stmts . JUMP_BACK
for_block ::= l_stmts_opt . _come_froms JUMP_BACK
for_block ::= l_stmts_opt . come_from_loops JUMP_BACK
for_block ::= l_stmts_opt \e__come_froms . JUMP_BACK
for_block ::= l_stmts_opt \e_come_from_loops . JUMP_BACK
for_block ::= l_stmts_opt _come_froms . JUMP_BACK
forelselaststmt38 ::= expr . get_for_iter store for_block POP_BLOCK else_suitec
forelselaststmt38 ::= expr get_for_iter . store for_block POP_BLOCK else_suitec
forelselaststmt38 ::= expr get_for_iter store . for_block POP_BLOCK else_suitec
forelselaststmt38 ::= expr get_for_iter store for_block . POP_BLOCK else_suitec
forelselaststmtl38 ::= expr . get_for_iter store for_block POP_BLOCK else_suitel
forelselaststmtl38 ::= expr get_for_iter . store for_block POP_BLOCK else_suitel
forelselaststmtl38 ::= expr get_for_iter store . for_block POP_BLOCK else_suitel
forelselaststmtl38 ::= expr get_for_iter store for_block . POP_BLOCK else_suitel
forelsestmt38 ::= expr . get_for_iter store for_block JUMP_BACK \e__come_froms else_suite
forelsestmt38 ::= expr . get_for_iter store for_block JUMP_BACK _come_froms else_suite
forelsestmt38 ::= expr . get_for_iter store for_block POP_BLOCK else_suite
forelsestmt38 ::= expr get_for_iter . store for_block JUMP_BACK \e__come_froms else_suite
forelsestmt38 ::= expr get_for_iter . store for_block JUMP_BACK _come_froms else_suite
forelsestmt38 ::= expr get_for_iter . store for_block POP_BLOCK else_suite
forelsestmt38 ::= expr get_for_iter store . for_block JUMP_BACK \e__come_froms else_suite
forelsestmt38 ::= expr get_for_iter store . for_block JUMP_BACK _come_froms else_suite
forelsestmt38 ::= expr get_for_iter store . for_block POP_BLOCK else_suite
forelsestmt38 ::= expr get_for_iter store for_block . JUMP_BACK \e__come_froms else_suite
forelsestmt38 ::= expr get_for_iter store for_block . JUMP_BACK _come_froms else_suite
forelsestmt38 ::= expr get_for_iter store for_block . POP_BLOCK else_suite
get_for_iter ::= GET_ITER . _come_froms FOR_ITER
get_for_iter ::= GET_ITER \e__come_froms . FOR_ITER
get_for_iter ::= GET_ITER _come_froms . FOR_ITER
get_for_iter ::= GET_ITER _come_froms FOR_ITER . 
get_iter ::= expr . GET_ITER
get_iter ::= expr GET_ITER . 
if_exp ::= expr . jmp_false expr jf_cf expr COME_FROM
if_exp ::= expr . jmp_false expr jump_absolute_else expr
if_exp ::= expr jmp_false . expr jf_cf expr COME_FROM
if_exp ::= expr jmp_false . expr jump_absolute_else expr
if_exp ::= expr jmp_false expr . jf_cf expr COME_FROM
if_exp ::= expr jmp_false expr . jump_absolute_else expr
if_exp37 ::= expr . expr jf_cfs expr COME_FROM
if_exp37 ::= expr expr . jf_cfs expr COME_FROM
if_exp_37b ::= expr . jmp_false expr POP_JUMP_IF_FALSE jump_forward_else expr
if_exp_37b ::= expr jmp_false . expr POP_JUMP_IF_FALSE jump_forward_else expr
if_exp_37b ::= expr jmp_false expr . POP_JUMP_IF_FALSE jump_forward_else expr
if_exp_37b ::= expr jmp_false expr POP_JUMP_IF_FALSE . jump_forward_else expr
if_exp_lambda ::= expr . jmp_false expr return_if_lambda return_stmt_lambda LAMBDA_MARKER
if_exp_lambda ::= expr jmp_false . expr return_if_lambda return_stmt_lambda LAMBDA_MARKER
if_exp_lambda ::= expr jmp_false expr . return_if_lambda return_stmt_lambda LAMBDA_MARKER
if_exp_not ::= expr . jmp_true expr jump_forward_else expr COME_FROM
if_exp_not_lambda ::= expr . jmp_true expr return_if_lambda return_stmt_lambda LAMBDA_MARKER
if_exp_true ::= expr . JUMP_FORWARD expr COME_FROM
ifelsestmt ::= testexpr . c_stmts come_froms else_suite come_froms
ifelsestmt ::= testexpr . c_stmts_opt JUMP_FORWARD else_suite \e__come_froms
ifelsestmt ::= testexpr . c_stmts_opt JUMP_FORWARD else_suite \e_opt_come_from_except
ifelsestmt ::= testexpr . c_stmts_opt JUMP_FORWARD else_suite _come_froms
ifelsestmt ::= testexpr . c_stmts_opt JUMP_FORWARD else_suite opt_come_from_except
ifelsestmt ::= testexpr . c_stmts_opt jf_cfs else_suite \e_opt_come_from_except
ifelsestmt ::= testexpr . c_stmts_opt jf_cfs else_suite opt_come_from_except
ifelsestmt ::= testexpr . c_stmts_opt jump_forward_else else_suite \e__come_froms
ifelsestmt ::= testexpr . c_stmts_opt jump_forward_else else_suite _come_froms
ifelsestmt ::= testexpr . stmts jf_cfs \e_else_suite_opt \e_opt_come_from_except
ifelsestmt ::= testexpr . stmts jf_cfs \e_else_suite_opt opt_come_from_except
ifelsestmt ::= testexpr . stmts jf_cfs else_suite_opt \e_opt_come_from_except
ifelsestmt ::= testexpr . stmts jf_cfs else_suite_opt opt_come_from_except
ifelsestmt ::= testexpr \e_c_stmts_opt . JUMP_FORWARD else_suite \e__come_froms
ifelsestmt ::= testexpr \e_c_stmts_opt . JUMP_FORWARD else_suite \e_opt_come_from_except
ifelsestmt ::= testexpr \e_c_stmts_opt . JUMP_FORWARD else_suite _come_froms
ifelsestmt ::= testexpr \e_c_stmts_opt . JUMP_FORWARD else_suite opt_come_from_except
ifelsestmt ::= testexpr \e_c_stmts_opt . jf_cfs else_suite \e_opt_come_from_except
ifelsestmt ::= testexpr \e_c_stmts_opt . jf_cfs else_suite opt_come_from_except
ifelsestmt ::= testexpr \e_c_stmts_opt . jump_forward_else else_suite \e__come_froms
ifelsestmt ::= testexpr \e_c_stmts_opt . jump_forward_else else_suite _come_froms
ifelsestmt ::= testexpr c_stmts . come_froms else_suite come_froms
ifelsestmt ::= testexpr c_stmts come_froms . else_suite come_froms
ifelsestmt ::= testexpr c_stmts come_froms else_suite . come_froms
ifelsestmt ::= testexpr c_stmts come_froms else_suite come_froms . 
ifelsestmt ::= testexpr c_stmts_opt . JUMP_FORWARD else_suite \e__come_froms
ifelsestmt ::= testexpr c_stmts_opt . JUMP_FORWARD else_suite \e_opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt . JUMP_FORWARD else_suite _come_froms
ifelsestmt ::= testexpr c_stmts_opt . JUMP_FORWARD else_suite opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt . jf_cfs else_suite \e_opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt . jf_cfs else_suite opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt . jump_forward_else else_suite \e__come_froms
ifelsestmt ::= testexpr c_stmts_opt . jump_forward_else else_suite _come_froms
ifelsestmt ::= testexpr c_stmts_opt JUMP_FORWARD . else_suite \e__come_froms
ifelsestmt ::= testexpr c_stmts_opt JUMP_FORWARD . else_suite \e_opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt JUMP_FORWARD . else_suite _come_froms
ifelsestmt ::= testexpr c_stmts_opt JUMP_FORWARD . else_suite opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt jf_cfs . else_suite \e_opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt jf_cfs . else_suite opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt jf_cfs else_suite . opt_come_from_except
ifelsestmt ::= testexpr c_stmts_opt jf_cfs else_suite \e_opt_come_from_except . 
ifelsestmt ::= testexpr c_stmts_opt jf_cfs else_suite opt_come_from_except . 
ifelsestmt ::= testexpr c_stmts_opt jump_forward_else . else_suite \e__come_froms
ifelsestmt ::= testexpr c_stmts_opt jump_forward_else . else_suite _come_froms
ifelsestmt ::= testexpr c_stmts_opt jump_forward_else else_suite . _come_froms
ifelsestmt ::= testexpr c_stmts_opt jump_forward_else else_suite \e__come_froms . 
ifelsestmt ::= testexpr c_stmts_opt jump_forward_else else_suite _come_froms . 
ifelsestmt ::= testexpr stmts . jf_cfs \e_else_suite_opt \e_opt_come_from_except
ifelsestmt ::= testexpr stmts . jf_cfs \e_else_suite_opt opt_come_from_except
ifelsestmt ::= testexpr stmts . jf_cfs else_suite_opt \e_opt_come_from_except
ifelsestmt ::= testexpr stmts . jf_cfs else_suite_opt opt_come_from_except
ifelsestmt ::= testexpr stmts jf_cfs . else_suite_opt \e_opt_come_from_except
ifelsestmt ::= testexpr stmts jf_cfs . else_suite_opt opt_come_from_except
ifelsestmt ::= testexpr stmts jf_cfs \e_else_suite_opt . opt_come_from_except
ifelsestmt ::= testexpr stmts jf_cfs \e_else_suite_opt \e_opt_come_from_except . 
ifelsestmt ::= testexpr stmts jf_cfs \e_else_suite_opt opt_come_from_except . 
ifelsestmt ::= testexpr stmts jf_cfs else_suite_opt . opt_come_from_except
ifelsestmt ::= testexpr stmts jf_cfs else_suite_opt \e_opt_come_from_except . 
ifelsestmt ::= testexpr stmts jf_cfs else_suite_opt opt_come_from_except . 
ifelsestmtc ::= testexpr . c_stmts_opt JUMP_ABSOLUTE else_suitec
ifelsestmtc ::= testexpr . c_stmts_opt JUMP_FORWARD else_suitec
ifelsestmtc ::= testexpr . c_stmts_opt jump_absolute_else else_suitec
ifelsestmtc ::= testexpr \e_c_stmts_opt . JUMP_ABSOLUTE else_suitec
ifelsestmtc ::= testexpr \e_c_stmts_opt . JUMP_FORWARD else_suitec
ifelsestmtc ::= testexpr \e_c_stmts_opt . jump_absolute_else else_suitec
ifelsestmtc ::= testexpr c_stmts_opt . JUMP_ABSOLUTE else_suitec
ifelsestmtc ::= testexpr c_stmts_opt . JUMP_FORWARD else_suitec
ifelsestmtc ::= testexpr c_stmts_opt . jump_absolute_else else_suitec
ifelsestmtl ::= testexpr . c_stmts cf_pt else_suite
ifelsestmtl ::= testexpr . c_stmts_opt cf_jf_else else_suitel
ifelsestmtl ::= testexpr . c_stmts_opt cf_jump_back else_suitel
ifelsestmtl ::= testexpr . c_stmts_opt jb_cfs else_suitel
ifelsestmtl ::= testexpr . c_stmts_opt jb_cfs else_suitel JUMP_BACK come_froms
ifelsestmtl ::= testexpr . c_stmts_opt jb_else else_suitel
ifelsestmtl ::= testexpr . c_stmts_opt jump_forward_else else_suitec
ifelsestmtl ::= testexpr \e_c_stmts_opt . cf_jf_else else_suitel
ifelsestmtl ::= testexpr \e_c_stmts_opt . cf_jump_back else_suitel
ifelsestmtl ::= testexpr \e_c_stmts_opt . jb_cfs else_suitel
ifelsestmtl ::= testexpr \e_c_stmts_opt . jb_cfs else_suitel JUMP_BACK come_froms
ifelsestmtl ::= testexpr \e_c_stmts_opt . jb_else else_suitel
ifelsestmtl ::= testexpr \e_c_stmts_opt . jump_forward_else else_suitec
ifelsestmtl ::= testexpr c_stmts . cf_pt else_suite
ifelsestmtl ::= testexpr c_stmts_opt . cf_jf_else else_suitel
ifelsestmtl ::= testexpr c_stmts_opt . cf_jump_back else_suitel
ifelsestmtl ::= testexpr c_stmts_opt . jb_cfs else_suitel
ifelsestmtl ::= testexpr c_stmts_opt . jb_cfs else_suitel JUMP_BACK come_froms
ifelsestmtl ::= testexpr c_stmts_opt . jb_else else_suitel
ifelsestmtl ::= testexpr c_stmts_opt . jump_forward_else else_suitec
ifelsestmtl ::= testexpr c_stmts_opt jump_forward_else . else_suitec
ifelsestmtl ::= testexpr c_stmts_opt jump_forward_else else_suitec . 
ifelsestmtr ::= testexpr . return_if_stmts returns
iflaststmt ::= testexpr . c_stmts
iflaststmt ::= testexpr . c_stmts JUMP_ABSOLUTE
iflaststmt ::= testexpr . c_stmts_opt JUMP_FORWARD
iflaststmt ::= testexpr \e_c_stmts_opt . JUMP_FORWARD
iflaststmt ::= testexpr c_stmts . 
iflaststmt ::= testexpr c_stmts . JUMP_ABSOLUTE
iflaststmt ::= testexpr c_stmts_opt . JUMP_FORWARD
iflaststmtl ::= testexpr . c_stmts
iflaststmtl ::= testexpr . c_stmts JUMP_BACK
iflaststmtl ::= testexpr . c_stmts JUMP_BACK COME_FROM_LOOP
iflaststmtl ::= testexpr . c_stmts JUMP_BACK POP_BLOCK
iflaststmtl ::= testexpr c_stmts . 
iflaststmtl ::= testexpr c_stmts . JUMP_BACK
iflaststmtl ::= testexpr c_stmts . JUMP_BACK COME_FROM_LOOP
iflaststmtl ::= testexpr c_stmts . JUMP_BACK POP_BLOCK
ifpoplaststmtl ::= testexpr . POP_TOP \e_c_stmts_opt
ifpoplaststmtl ::= testexpr . POP_TOP c_stmts_opt
ifpoplaststmtl ::= testexpr POP_TOP . c_stmts_opt
ifpoplaststmtl ::= testexpr POP_TOP \e_c_stmts_opt . 
ifstmt ::= testexpr . _ifstmts_jump
ifstmt ::= testexpr _ifstmts_jump . 
ifstmtl ::= testexpr . _ifstmts_jumpl
ifstmtl ::= testexpr _ifstmts_jumpl . 
import ::= LOAD_CONST . LOAD_CONST alias
import_as37 ::= LOAD_CONST . LOAD_CONST importlist37 store POP_TOP
import_from ::= LOAD_CONST . LOAD_CONST IMPORT_NAME importlist POP_TOP
import_from ::= LOAD_CONST . LOAD_CONST importlist POP_TOP
import_from37 ::= LOAD_CONST . LOAD_CONST IMPORT_NAME_ATTR importlist37 POP_TOP
import_from_as37 ::= LOAD_CONST . LOAD_CONST import_from_attr37 store POP_TOP
import_from_star ::= LOAD_CONST . LOAD_CONST IMPORT_NAME IMPORT_STAR
import_from_star ::= LOAD_CONST . LOAD_CONST IMPORT_NAME_ATTR IMPORT_STAR
importmultiple ::= LOAD_CONST . LOAD_CONST alias imports_cont
inplace_op ::= INPLACE_ADD . 
jb_cfs ::= \e_come_from_opt . JUMP_BACK come_froms
jb_cfs ::= come_from_opt . JUMP_BACK come_froms
jf_cfs ::= JUMP_FORWARD . _come_froms
jf_cfs ::= JUMP_FORWARD \e__come_froms . 
jf_cfs ::= JUMP_FORWARD _come_froms . 
jmp_false ::= POP_JUMP_IF_FALSE . 
jump_absolute_else ::= come_froms . _jump COME_FROM
jump_forward_else ::= JUMP_FORWARD . 
jump_forward_else ::= JUMP_FORWARD . COME_FROM
jump_forward_else ::= JUMP_FORWARD . ELSE
jump_forward_else ::= JUMP_FORWARD COME_FROM . 
kvlist_0 ::= BUILD_MAP_0 . 
l_stmts ::= _stmts . 
l_stmts ::= _stmts . lastl_stmt
l_stmts ::= _stmts lastl_stmt . 
l_stmts ::= l_stmts . lstmt
l_stmts ::= l_stmts lstmt . 
l_stmts ::= lastl_stmt . 
l_stmts ::= lastl_stmt . come_froms l_stmts
l_stmts ::= lastl_stmt come_froms . l_stmts
l_stmts ::= lastl_stmt come_froms l_stmts . 
l_stmts ::= lstmt . 
l_stmts_opt ::= l_stmts . 
lambda_body ::= expr . LOAD_LAMBDA LOAD_STR MAKE_FUNCTION_1
lastl_stmt ::= ifelsestmtl . 
lastl_stmt ::= iflaststmtl . 
lastl_stmt ::= ifpoplaststmtl . 
lc_setup_finally ::= LOAD_CONST . SETUP_FINALLY
list ::= BUILD_LIST_0 . 
list_comp ::= BUILD_LIST_0 . list_iter
list_unpack ::= BUILD_LIST_0 . expr LIST_EXTEND
lstmt ::= stmt . 
mkfunc ::= expr . LOAD_CODE LOAD_STR MAKE_FUNCTION_1
mkfuncdeco ::= expr . mkfuncdeco CALL_FUNCTION_1
mkfuncdeco ::= expr . mkfuncdeco0 CALL_FUNCTION_1
named_expr ::= expr . DUP_TOP store
opt_come_from_except ::= _come_froms . 
or ::= and . jitop_come_from_expr COME_FROM
pop_ex_return ::= return_expr . ROT_FOUR POP_EXCEPT RETURN_VALUE
pop_return ::= POP_TOP . return_expr RETURN_VALUE
pop_return ::= POP_TOP return_expr . RETURN_VALUE
popb_return ::= return_expr . POP_BLOCK RETURN_VALUE
pos_arg ::= expr . 
raise_stmt1 ::= expr . RAISE_VARARGS_1
ret_and ::= expr . JUMP_IF_FALSE_OR_POP return_expr_or_cond COME_FROM
ret_or ::= expr . JUMP_IF_TRUE_OR_POP return_expr_or_cond COME_FROM
return ::= return_expr . RETURN_END_IF
return ::= return_expr . RETURN_VALUE
return ::= return_expr . RETURN_VALUE COME_FROM
return ::= return_expr . discard_tops RETURN_VALUE
return_expr ::= expr . 
return_expr_lambda ::= return_expr . RETURN_VALUE_LAMBDA
return_expr_lambda ::= return_expr . RETURN_VALUE_LAMBDA LAMBDA_MARKER
return_if_stmt ::= return_expr . RETURN_END_IF
return_if_stmt ::= return_expr . RETURN_END_IF POP_BLOCK
return_if_stmts ::= _stmts . return_if_stmt \e__come_froms
return_if_stmts ::= _stmts . return_if_stmt _come_froms
returns ::= _stmts . return
returns ::= _stmts . return_if_stmt
slice2 ::= expr . expr BUILD_SLICE_2
slice2 ::= expr expr . BUILD_SLICE_2
sstmt ::= sstmt . RETURN_LAST
sstmt ::= stmt . 
stmt ::= assign . 
stmt ::= aug_assign1 . 
stmt ::= call_stmt . 
stmt ::= expr_stmt . 
stmt ::= for38 . 
stmt ::= ifelsestmt . 
stmt ::= ifstmt . 
stmt ::= ifstmtl . 
stmts ::= sstmt . 
stmts ::= stmts . sstmt
stmts ::= stmts sstmt . 
store ::= STORE_FAST . 
store ::= expr . STORE_ATTR
store ::= unpack . 
store_subscript ::= expr . expr STORE_SUBSCR
store_subscript ::= expr expr . STORE_SUBSCR
subscript ::= expr . expr BINARY_SUBSCR
subscript ::= expr expr . BINARY_SUBSCR
subscript ::= expr expr BINARY_SUBSCR . 
subscript2 ::= expr . expr DUP_TOP_TWO BINARY_SUBSCR
subscript2 ::= expr expr . DUP_TOP_TWO BINARY_SUBSCR
suite_stmts ::= _stmts . 
testexpr ::= testfalse . 
testexpr_cf ::= testexpr . come_froms
testfalse ::= expr . jmp_false
testfalse ::= expr jmp_false . 
testfalse_not_and ::= and . jmp_true come_froms
testfalse_not_and ::= expr . jmp_false expr jmp_true COME_FROM
testfalse_not_and ::= expr jmp_false . expr jmp_true COME_FROM
testfalse_not_and ::= expr jmp_false expr . jmp_true COME_FROM
testfalse_not_or ::= expr . jmp_false expr jmp_false COME_FROM
testfalse_not_or ::= expr jmp_false . expr jmp_false COME_FROM
testfalse_not_or ::= expr jmp_false expr . jmp_false COME_FROM
testfalse_not_or ::= expr jmp_false expr jmp_false . COME_FROM
testfalsel ::= expr . jmp_true
testtrue ::= expr . jmp_true
tryfinally38astmt ::= LOAD_CONST . SETUP_FINALLY \e_suite_stmts_opt POP_BLOCK BEGIN_FINALLY COME_FROM_FINALLY POP_FINALLY POP_TOP \e_suite_stmts_opt END_FINALLY POP_TOP
tryfinally38astmt ::= LOAD_CONST . SETUP_FINALLY \e_suite_stmts_opt POP_BLOCK BEGIN_FINALLY COME_FROM_FINALLY POP_FINALLY POP_TOP suite_stmts_opt END_FINALLY POP_TOP
tryfinally38astmt ::= LOAD_CONST . SETUP_FINALLY suite_stmts_opt POP_BLOCK BEGIN_FINALLY COME_FROM_FINALLY POP_FINALLY POP_TOP \e_suite_stmts_opt END_FINALLY POP_TOP
tryfinally38astmt ::= LOAD_CONST . SETUP_FINALLY suite_stmts_opt POP_BLOCK BEGIN_FINALLY COME_FROM_FINALLY POP_FINALLY POP_TOP suite_stmts_opt END_FINALLY POP_TOP
tuple ::= expr . expr BUILD_TUPLE_2
tuple ::= expr . expr expr BUILD_TUPLE_3
tuple ::= expr . expr expr expr BUILD_TUPLE_4
tuple ::= expr . expr expr expr expr BUILD_TUPLE_5
tuple ::= expr expr . BUILD_TUPLE_2
tuple ::= expr expr . expr BUILD_TUPLE_3
tuple ::= expr expr . expr expr BUILD_TUPLE_4
tuple ::= expr expr . expr expr expr BUILD_TUPLE_5
tuple ::= expr expr expr . BUILD_TUPLE_3
tuple ::= expr expr expr . expr BUILD_TUPLE_4
tuple ::= expr expr expr . expr expr BUILD_TUPLE_5
tuple ::= expr expr expr expr . BUILD_TUPLE_4
tuple ::= expr expr expr expr . expr BUILD_TUPLE_5
tuple ::= expr expr expr expr expr . BUILD_TUPLE_5
unary_not ::= expr . UNARY_NOT
unary_op ::= expr . unary_operator
unpack ::= UNPACK_SEQUENCE_2 . store store
unpack ::= UNPACK_SEQUENCE_2 store . store
unpack ::= UNPACK_SEQUENCE_2 store store . 
unpack ::= UNPACK_SEQUENCE_4 . store store store store
unpack ::= UNPACK_SEQUENCE_4 store . store store store
unpack ::= UNPACK_SEQUENCE_4 store store . store store
unpack ::= UNPACK_SEQUENCE_4 store store store . store
unpack ::= UNPACK_SEQUENCE_4 store store store store . 
unpack ::= UNPACK_SEQUENCE_5 . store store store store store
unpack ::= UNPACK_SEQUENCE_5 store . store store store store
unpack ::= UNPACK_SEQUENCE_5 store store . store store store
unpack ::= UNPACK_SEQUENCE_5 store store store . store store
unpack ::= UNPACK_SEQUENCE_5 store store store store . store
unpack ::= UNPACK_SEQUENCE_5 store store store store store . 
while1stmt ::= \e__come_froms . l_stmts COME_FROM JUMP_BACK COME_FROM_LOOP
while1stmt ::= \e__come_froms l_stmts . COME_FROM JUMP_BACK COME_FROM_LOOP
while1stmt ::= \e__come_froms l_stmts COME_FROM . JUMP_BACK COME_FROM_LOOP
while1stmt ::= _come_froms . l_stmts COME_FROM JUMP_BACK COME_FROM_LOOP
while1stmt ::= _come_froms l_stmts . COME_FROM JUMP_BACK COME_FROM_LOOP
while1stmt ::= _come_froms l_stmts COME_FROM . JUMP_BACK COME_FROM_LOOP
whileTruestmt ::= \e__come_froms . l_stmts JUMP_BACK POP_BLOCK
whileTruestmt ::= \e__come_froms l_stmts . JUMP_BACK POP_BLOCK
whileTruestmt ::= _come_froms . l_stmts JUMP_BACK POP_BLOCK
whileTruestmt ::= _come_froms l_stmts . JUMP_BACK POP_BLOCK
whileTruestmt38 ::= \e__come_froms . l_stmts JUMP_BACK
whileTruestmt38 ::= \e__come_froms . l_stmts JUMP_BACK COME_FROM_EXCEPT_CLAUSE
whileTruestmt38 ::= \e__come_froms . pass JUMP_BACK
whileTruestmt38 ::= \e__come_froms \e_pass . JUMP_BACK
whileTruestmt38 ::= \e__come_froms l_stmts . JUMP_BACK
whileTruestmt38 ::= \e__come_froms l_stmts . JUMP_BACK COME_FROM_EXCEPT_CLAUSE
whileTruestmt38 ::= _come_froms . l_stmts JUMP_BACK
whileTruestmt38 ::= _come_froms . l_stmts JUMP_BACK COME_FROM_EXCEPT_CLAUSE
whileTruestmt38 ::= _come_froms . pass JUMP_BACK
whileTruestmt38 ::= _come_froms \e_pass . JUMP_BACK
whileTruestmt38 ::= _come_froms l_stmts . JUMP_BACK
whileTruestmt38 ::= _come_froms l_stmts . JUMP_BACK COME_FROM_EXCEPT_CLAUSE
whilestmt38 ::= \e__come_froms . testexpr \e_l_stmts_opt COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms . testexpr \e_l_stmts_opt JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms . testexpr \e_l_stmts_opt JUMP_BACK come_froms
whilestmt38 ::= \e__come_froms . testexpr l_stmts JUMP_BACK
whilestmt38 ::= \e__come_froms . testexpr l_stmts come_froms
whilestmt38 ::= \e__come_froms . testexpr l_stmts_opt COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms . testexpr l_stmts_opt JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms . testexpr l_stmts_opt JUMP_BACK come_froms
whilestmt38 ::= \e__come_froms . testexpr returns POP_BLOCK
whilestmt38 ::= \e__come_froms testexpr . l_stmts JUMP_BACK
whilestmt38 ::= \e__come_froms testexpr . l_stmts come_froms
whilestmt38 ::= \e__come_froms testexpr . l_stmts_opt COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms testexpr . l_stmts_opt JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms testexpr . l_stmts_opt JUMP_BACK come_froms
whilestmt38 ::= \e__come_froms testexpr . returns POP_BLOCK
whilestmt38 ::= \e__come_froms testexpr \e_l_stmts_opt . COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms testexpr \e_l_stmts_opt . JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms testexpr \e_l_stmts_opt . JUMP_BACK come_froms
whilestmt38 ::= \e__come_froms testexpr l_stmts . JUMP_BACK
whilestmt38 ::= \e__come_froms testexpr l_stmts . come_froms
whilestmt38 ::= \e__come_froms testexpr l_stmts come_froms . 
whilestmt38 ::= \e__come_froms testexpr l_stmts_opt . COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms testexpr l_stmts_opt . JUMP_BACK POP_BLOCK
whilestmt38 ::= \e__come_froms testexpr l_stmts_opt . JUMP_BACK come_froms
whilestmt38 ::= \e__come_froms testexpr l_stmts_opt COME_FROM . JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms . testexpr \e_l_stmts_opt COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms . testexpr \e_l_stmts_opt JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms . testexpr \e_l_stmts_opt JUMP_BACK come_froms
whilestmt38 ::= _come_froms . testexpr l_stmts JUMP_BACK
whilestmt38 ::= _come_froms . testexpr l_stmts come_froms
whilestmt38 ::= _come_froms . testexpr l_stmts_opt COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms . testexpr l_stmts_opt JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms . testexpr l_stmts_opt JUMP_BACK come_froms
whilestmt38 ::= _come_froms . testexpr returns POP_BLOCK
whilestmt38 ::= _come_froms testexpr . l_stmts JUMP_BACK
whilestmt38 ::= _come_froms testexpr . l_stmts come_froms
whilestmt38 ::= _come_froms testexpr . l_stmts_opt COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms testexpr . l_stmts_opt JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms testexpr . l_stmts_opt JUMP_BACK come_froms
whilestmt38 ::= _come_froms testexpr . returns POP_BLOCK
whilestmt38 ::= _come_froms testexpr \e_l_stmts_opt . COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms testexpr \e_l_stmts_opt . JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms testexpr \e_l_stmts_opt . JUMP_BACK come_froms
whilestmt38 ::= _come_froms testexpr l_stmts . JUMP_BACK
whilestmt38 ::= _come_froms testexpr l_stmts . come_froms
whilestmt38 ::= _come_froms testexpr l_stmts come_froms . 
whilestmt38 ::= _come_froms testexpr l_stmts_opt . COME_FROM JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms testexpr l_stmts_opt . JUMP_BACK POP_BLOCK
whilestmt38 ::= _come_froms testexpr l_stmts_opt . JUMP_BACK come_froms
whilestmt38 ::= _come_froms testexpr l_stmts_opt COME_FROM . JUMP_BACK POP_BLOCK
yield ::= expr . YIELD_VALUE
yield_from ::= expr . GET_YIELD_FROM_ITER LOAD_CONST YIELD_FROM
Instruction context:
   
 L. 407       350  POP_TOP          
->           352_354  JUMP_FORWARD        668  'to 668'
               356_0  COME_FROM           346  '346'
               356_1  COME_FROM           306  '306'
               356_2  COME_FROM           296  '296'
from PIL import Image
import os, time, numpy as np, torch
import torch.nn.functional as F
import data
from utils import frame_utils
from utils.flow_viz import save_vis_flow_tofile
from utils.utils import InputPadder, compute_out_of_boundary_mask
from glob import glob
from gmflow.geometry import forward_backward_consistency_check

@torch.no_grad()
def create_sintel_submission(model, output_path='sintel_submission', padding_factor=8, save_vis_flow=False, no_save_flo=False, attn_splits_list=None, corr_radius_list=None, prop_radius_list=None):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ('clean', 'final'):
        test_dataset = data.MpiSintel(split="test", aug_params=None, dstype=dstype)
        flow_prev, sequence_prev = (None, None)
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            padder = InputPadder((image1.shape), padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
            results_dict = model(image1, image2, attn_splits_list=attn_splits_list,
              corr_radius_list=corr_radius_list,
              prop_radius_list=prop_radius_list)
            flow_pr = results_dict["flow_preds"][-1]
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, "frame%04d.flo" % (frame + 1))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not no_save_flo:
                frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence
            if save_vis_flow:
                vis_flow_file = output_file.replace(".flo", ".png")
                save_vis_flow_tofile(flow, vis_flow_file)


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission', padding_factor=8, save_vis_flow=False, attn_splits_list=None, corr_radius_list=None, prop_radius_list=None):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = data.KITTI(split="testing", aug_params=None)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder((image1.shape), mode="kitti", padding_factor=padding_factor)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        results_dict = model(image1, image2, attn_splits_list=attn_splits_list,
          corr_radius_list=corr_radius_list,
          prop_radius_list=prop_radius_list)
        flow_pr = results_dict["flow_preds"][-1]
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        output_filename = os.path.join(output_path, frame_id)
        if save_vis_flow:
            vis_flow_file = output_filename
            save_vis_flow_tofile(flow, vis_flow_file)
        else:
            frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_fppdic(model, with_speed_metric=False, attn_splits_list=False, corr_radius_list=False, prop_radius_list=False):
    """最简单的验证代码，在2025年8月19日这个时间点上其他的验证方法也不会，在fppdic数据集上进行验证"""
    model.eval()
    epe_list = []
    results = {}
    val_dataset = data.BlenderEXRDataset(split="val")
    print("Number of validation image pairs: %d" % len(val_dataset))
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        results_dict = model(image1, image2, attn_splits_list=attn_splits_list,
          corr_radius_list=corr_radius_list,
          prop_radius_list=prop_radius_list)
        flow_pr = results_dict["flow_preds"][-1]
        assert flow_pr.size()[(-2)[:None]] == flow_gt.size()[(-2)[:None]]
        epe = torch.sum(((flow_pr[0].cpu() - flow_gt) ** 2), dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
    else:
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all > 1)
        px3 = np.mean(epe_all > 3)
        px5 = np.mean(epe_all > 5)
        print("Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
        results["chairs_epe"] = epe
        results["chairs_1px"] = px1
        results["chairs_3px"] = px3
        results["chairs_5px"] = px5
        return results


@torch.no_grad()
def validate_chairs(model, with_speed_metric=False, attn_splits_list=False, corr_radius_list=False, prop_radius_list=False):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    results = {}
    if with_speed_metric:
        s0_10_list = []
        s10_40_list = []
        s40plus_list = []
    val_dataset = data.FlyingChairs(split="validation")
    print("Number of validation image pairs: %d" % len(val_dataset))
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        results_dict = model(image1, image2, attn_splits_list=attn_splits_list,
          corr_radius_list=corr_radius_list,
          prop_radius_list=prop_radius_list)
        flow_pr = results_dict["flow_preds"][-1]
        assert flow_pr.size()[(-2)[:None]] == flow_gt.size()[(-2)[:None]]
        epe = torch.sum(((flow_pr[0].cpu() - flow_gt) ** 2), dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
        if with_speed_metric:
            flow_gt_speed = torch.sum((flow_gt ** 2), dim=0).sqrt()
            valid_mask = flow_gt_speed < 10
            if valid_mask.max() > 0:
                s0_10_list.append(epe[valid_mask].cpu().numpy())
            valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
            if valid_mask.max() > 0:
                s10_40_list.append(epe[valid_mask].cpu().numpy())
            valid_mask = flow_gt_speed > 40
            if valid_mask.max() > 0:
                s40plus_list.append(epe[valid_mask].cpu().numpy())
            epe_all = np.concatenate(epe_list)
            epe = np.mean(epe_all)
            px1 = np.mean(epe_all > 1)
            px3 = np.mean(epe_all > 3)
            px5 = np.mean(epe_all > 5)
            print("Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
            results["chairs_epe"] = epe
            results["chairs_1px"] = px1
            results["chairs_3px"] = px3
            results["chairs_5px"] = px5
            if with_speed_metric:
                s0_10 = np.mean(np.concatenate(s0_10_list))
                s10_40 = np.mean(np.concatenate(s10_40_list))
                s40plus = np.mean(np.concatenate(s40plus_list))
                print("Validation Chairs s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
                 s0_10,
                 s10_40,
                 s40plus))
                results["chairs_s0_10"] = s0_10
                results["chairs_s10_40"] = s10_40
                results["chairs_s40+"] = s40plus
        return results


@torch.no_grad()
def validate_things(model, padding_factor=8, with_speed_metric=False, max_val_flow=400, val_things_clean_only=True, attn_splits_list=False, corr_radius_list=False, prop_radius_list=False):
    """ Peform validation using the Things (test) split """
    model.eval()
    results = {}
    for dstype in ('frames_cleanpass', 'frames_finalpass'):
        if val_things_clean_only and dstype == "frames_finalpass":
            pass
        else:
            val_dataset = data.FlyingThings3D(dstype=dstype, test_set=True, validate_subset=True)
            print("Number of validation image pairs: %d" % len(val_dataset))
            epe_list = []
            if with_speed_metric:
                s0_10_list = []
                s10_40_list = []
                s40plus_list = []
            for val_id in range(len(val_dataset)):
                image1, image2, flow_gt, valid_gt = val_dataset[val_id]
                image1 = image1[None].cuda()
                image2 = image2[None].cuda()
                padder = InputPadder((image1.shape), padding_factor=padding_factor)
                image1, image2 = padder.pad(image1, image2)
                results_dict = model(image1, image2, attn_splits_list=attn_splits_list,
                  corr_radius_list=corr_radius_list,
                  prop_radius_list=prop_radius_list)
                flow_pr = results_dict["flow_preds"][-1]
                flow = padder.unpad(flow_pr[0]).cpu()
                flow_gt_speed = torch.sum((flow_gt ** 2), dim=0).sqrt()
                valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
                valid_gt = valid_gt.contiguous()
                epe = torch.sum(((flow - flow_gt) ** 2), dim=0).sqrt()
                val = valid_gt >= 0.5
                epe_list.append(epe[val].cpu().numpy())
                if with_speed_metric:
                    valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)
                    if valid_mask.max() > 0:
                        s0_10_list.append(epe[valid_mask].cpu().numpy())
                    valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                    if valid_mask.max() > 0:
                        s10_40_list.append(epe[valid_mask].cpu().numpy())
                    valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                    if valid_mask.max() > 0:
                        s40plus_list.append(epe[valid_mask].cpu().numpy())
                    epe_list = np.mean(np.concatenate(epe_list))
                    epe = np.mean(epe_list)
                    if dstype == "frames_cleanpass":
                        dstype = "things_clean"
                    if dstype == "frames_finalpass":
                        dstype = "things_final"
                    print("Validation Things test set (%s) EPE: %.3f" % (dstype, epe))
                    results[dstype + "_epe"] = epe
                    if with_speed_metric:
                        s0_10 = np.mean(np.concatenate(s0_10_list))
                        s10_40 = np.mean(np.concatenate(s10_40_list))
                        s40plus = np.mean(np.concatenate(s40plus_list))
                        print("Validation Things test (%s) s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
                         dstype, s0_10,
                         s10_40,
                         s40plus))
                        results[dstype + "_s0_10"] = s0_10
                        results[dstype + "_s10_40"] = s10_40
                        results[dstype + "_s40+"] = s40plus
                return results


@torch.no_grad()
def validate_sintelParse error at or near `JUMP_FORWARD' instruction at offset 352_354


@torch.no_grad()
def validate_kitti(model, padding_factor=8, with_speed_metric=False, average_over_pixels=True, attn_splits_list=False, corr_radius_list=False, prop_radius_list=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = data.KITTI(split="training")
    print("Number of validation image pairs: %d" % len(val_dataset))
    out_list, epe_list = [], []
    results = {}
    if with_speed_metric:
        if average_over_pixels:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []
        else:
            s0_10_epe_sum = 0
            s0_10_valid_samples = 0
            s10_40_epe_sum = 0
            s10_40_valid_samples = 0
            s40plus_epe_sum = 0
            s40plus_valid_samples = 0
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder((image1.shape), mode="kitti", padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)
        results_dict = model(image1, image2, attn_splits_list=attn_splits_list,
          corr_radius_list=corr_radius_list,
          prop_radius_list=prop_radius_list)
        flow_pr = results_dict["flow_preds"][-1]
        flow = padder.unpad(flow_pr[0]).cpu()
        epe = torch.sum(((flow - flow_gt) ** 2), dim=0).sqrt()
        mag = torch.sum((flow_gt ** 2), dim=0).sqrt()
        if with_speed_metric:
            flow_gt_speed = mag
            if average_over_pixels:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())
                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())
                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())
        else:
            valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)
            if valid_mask.max() > 0:
                s0_10_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                s0_10_valid_samples += 1
            valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
            if valid_mask.max() > 0:
                s10_40_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                s10_40_valid_samples += 1
            valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
            if valid_mask.max() > 0:
                s40plus_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                s40plus_valid_samples += 1
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5
            out = ((epe > 3.0) & (epe / mag > 0.05)).float()
            if average_over_pixels:
                epe_list.append(epe[val].cpu().numpy())
            else:
                epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
    else:
        if average_over_pixels:
            epe_list = np.concatenate(epe_list)
        else:
            epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)
        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)
        print("Validation KITTI EPE: %.3f, F1-all: %.3f" % (epe, f1))
        results["kitti_epe"] = epe
        results["kitti_f1"] = f1
        if with_speed_metric:
            if average_over_pixels:
                s0_10 = np.mean(np.concatenate(s0_10_list))
                s10_40 = np.mean(np.concatenate(s10_40_list))
                s40plus = np.mean(np.concatenate(s40plus_list))
            else:
                s0_10 = s0_10_epe_sum / s0_10_valid_samples
                s10_40 = s10_40_epe_sum / s10_40_valid_samples
                s40plus = s40plus_epe_sum / s40plus_valid_samples
            print("Validation KITTI s0_10: %.3f, s10_40: %.3f, s40+: %.3f" % (
             s0_10,
             s10_40,
             s40plus))
            results["kitti_s0_10"] = s0_10
            results["kitti_s10_40"] = s10_40
            results["kitti_s40+"] = s40plus
        return results


@torch.no_grad()
def inference_on_dir(model, inference_dir, output_path='output', padding_factor=8, inference_size=None, paired_data=False, save_flo_flow=False, attn_splits_list=None, corr_radius_list=None, prop_radius_list=None, pred_bidir_flow=False, fwd_bwd_consistency_check=False):
    """ Inference on a directory """
    model.eval()
    if fwd_bwd_consistency_check:
        assert pred_bidir_flow
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filenames = sorted(glob(inference_dir + "/*"))
    print("%d images found" % len(filenames))
    stride = 2 if paired_data else 1
    if paired_data:
        assert len(filenames) % 2 == 0
    for test_id in range(0, len(filenames) - 1, stride):
        image1 = frame_utils.read_gen(filenames[test_id])
        image2 = frame_utils.read_gen(filenames[test_id + 1])
        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)
        if len(image1.shape) == 2:
            image1 = np.tile(image1[(Ellipsis, None)], (1, 1, 3))
            image2 = np.tile(image2[(Ellipsis, None)], (1, 1, 3))
        else:
            image1 = image1[(..., None[:3])]
            image2 = image2[(..., None[:3])]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        if inference_size is None:
            padder = InputPadder((image1.shape), padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        else:
            image1, image2 = image1[None].cuda(), image2[None].cuda()
        if inference_size is not None:
            if not isinstance(inference_size, list):
                assert isinstance(inference_size, tuple)
                ori_size = image1.shape[(-2)[:None]]
                image1 = F.interpolate(image1, size=inference_size, mode="bilinear", align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode="bilinear", align_corners=True)
            else:
                results_dict = model(image1, image2, attn_splits_list=attn_splits_list,
                  corr_radius_list=corr_radius_list,
                  prop_radius_list=prop_radius_list,
                  pred_bidir_flow=pred_bidir_flow)
                flow_pr = results_dict["flow_preds"][-1]
                if inference_size is not None:
                    flow_pr = F.interpolate(flow_pr, size=ori_size, mode="bilinear", align_corners=True)
                    flow_pr[(None[:None], 0)] = flow_pr[(None[:None], 0)] * ori_size[-1] / inference_size[-1]
                    flow_pr[(None[:None], 1)] = flow_pr[(None[:None], 1)] * ori_size[-2] / inference_size[-2]
                if inference_size is None:
                    flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
                else:
                    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()
            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[None[:-4]] + "_flow.png")
            save_vis_flow_tofile(flow, output_file)
            if pred_bidir_flow:
                if not flow_pr.size(0) == 2:
                    raise AssertionError
        elif inference_size is None:
            flow_bwd = padder.unpad(flow_pr[1]).permute(1, 2, 0).cpu().numpy()
        else:
            flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()
        output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[None[:-4]] + "_flow_bwd.png")
        save_vis_flow_tofile(flow_bwd, output_file)
        if fwd_bwd_consistency_check:
            if inference_size is None:
                fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)
                bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)
            else:
                fwd_flow = flow_pr[0].unsqueeze(0)
                bwd_flow = flow_pr[1].unsqueeze(0)
            fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)
            fwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[None[:-4]] + "_occ.png")
            bwd_occ_file = os.path.join(output_path, os.path.basename(filenames[test_id])[None[:-4]] + "_occ_bwd.png")
            Image.fromarray((fwd_occ[0].cpu().numpy() * 255.0).astype(np.uint8)).save(fwd_occ_file)
            Image.fromarray((bwd_occ[0].cpu().numpy() * 255.0).astype(np.uint8)).save(bwd_occ_file)
        if save_flo_flow:
            output_file = os.path.join(output_path, os.path.basename(filenames[test_id])[None[:-4]] + "_pred.flo")
            frame_utils.writeFlow(output_file, flow)
