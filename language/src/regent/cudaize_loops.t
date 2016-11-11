-- Copyright 2016 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Legion Loop Cudaizer
--
-- Attempts to cudaize the body of loops
--

local ast = require("regent/ast")
local ast_util = require("regent/ast_util")
local report = require("common/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")
local check_cudaizability = require("regent/check_cudaizability")

local c = std.c

local V = {}
V.__index = V
function V:__tostring() return "vector" end
setmetatable(V, V)
local S = {}
S.__index = S
function S:__tostring() return "scalar" end
setmetatable(S, S)
local C = {}
C.__index = C
function C:__tostring() return "vector-contiguous" end
setmetatable(C, C)

cudaize_loops = {}

cudaize_loops.pass_name = "cudaize_loops"

local context = {}
context.__index = context

function context:new_local_scope()
  local cx = {
    var_type = self.var_type:new_local_scope(),
    subst = self.subst:new_local_scope(),
    expr_type = self.expr_type,
    demanded = self.demanded,
    read_set = self.read_set,
    loop_symbol = self.loop_symbol,
  }
  return setmetatable(cx, context)
end

function context:new_global_scope(loop_symbol)
  local cx = {
    var_type = symbol_table:new_global_scope(),
    subst = symbol_table:new_global_scope(),
    expr_type = {},
    demanded = false,
    read_set = {},
    write_set = {},
    loop_symbol = loop_symbol,
  }
  return setmetatable(cx, context)
end

function context:assign(symbol, fact)
  self.var_type:insert(nil, symbol, fact)
end

function context:join(symbol, fact)
  local var_type = self.var_type
  local old_fact = var_type:safe_lookup(symbol)
  assert(old_fact)
  local new_fact = join(old_fact, fact)
  var_type:insert(nil, symbol, new_fact)
end

function context:lookup_expr_type(node)
  return self.expr_type[node]
end

function context:assign_expr_type(node, fact)
  self.expr_type[node] = fact
end

function context:join_expr_type(node, fact)
  self.expr_type[node] = join(fact, self.expr_type[node])
end

function context:report_error_when_demanded(node, error_msg)
  if self.demanded then report.error(node, error_msg) end
end

function context:add_substitution(from, to)
  self.subst:insert(nil, from, to)
end

function context:find_replacement(from)
  local to = self.subst:safe_lookup(from)
  assert(to)
  return to
end

-- visitor for each statement type
local cudaize_loops = {}

function cudaize_loops.block(node)
  local flattened_stats = terralib.newlist()
  local stats = node.stats:map(
    function(stat) return cudaize_loops.stat(stat) end)

  for _, s in ipairs(stats) do
    if terralib.islist(s) then
      flattened_stats:insertall(s)
    else
      flattened_stats:insert(s)
    end
  end
  return node {
    stats = flattened_stats
  }
end

function cudaize_loops.stat_if(node)
  return node {
    then_block = cudaize_loops.block(node.then_block),
    elseif_blocks = node.elseif_blocks:map(
      function(block) return cudaize_loops.stat_elseif(block) end),
    else_block = cudaize_loops.block(node.else_block),
  }
end

function cudaize_loops.stat_elseif(node)
  return node { block = cudaize_loops.block(node.block) }
end

function cudaize_loops.stat_while(node)
  return node { block = cudaize_loops.block(node.block) }
end

function cudaize_loops.stat_for_num(node)
  return node { block = cudaize_loops.block(node.block) }
end

function cudaize_loops.stat_for_list(node)
  local cx = context:new_global_scope(node.symbol)
  cx:assign(node.symbol, C)
  cx.demanded = true
  check_cudaizability.clear_reduction_node()

  local cudaizable = check_cudaizability.block(cx, node.block)
  local reduction = check_cudaizability.get_reduction_node()

  if cudaizable and reduction then
    return cudaize.stat_for_list(cx, node, reduction)
  else
    return node { block = cudaize_loops.block(node.block) }
  end
end

function cudaize_loops.stat_repeat(node)
  return node { block = cudaize_loops.block(node.block) }
end

function cudaize_loops.stat_must_epoch(node)
  return node { block = cudaize_loops.block(node.block) }
end

function cudaize_loops.stat_block(node)
  return node { block = cudaize_loops.block(node.block) }
end

function cudaize_loops.stat(node)
  if node:is(ast.typed.stat.If) then
    return cudaize_loops.stat_if(node)

  elseif node:is(ast.typed.stat.While) then
    return cudaize_loops.stat_while(node)

  elseif node:is(ast.typed.stat.ForNum) then
    return cudaize_loops.stat_for_num(node)

  elseif node:is(ast.typed.stat.ForList) then
    if std.is_bounded_type(node.symbol:gettype()) then
      return cudaize_loops.stat_for_list(node)
    else
      return node { block = cudaize_loops.block(node.block) }
    end

  elseif node:is(ast.typed.stat.Repeat) then
    return cudaize_loops.stat_repeat(node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return cudaize_loops.stat_must_epoch(node)

  elseif node:is(ast.typed.stat.Block) then
    return cudaize_loops.stat_block(node)

  elseif node:is(ast.typed.stat.IndexLaunchNum) then
    return node

  elseif node:is(ast.typed.stat.IndexLaunchList) then
    return node

  elseif node:is(ast.typed.stat.Var) then
    return node

  elseif node:is(ast.typed.stat.VarUnpack) then
    return node

  elseif node:is(ast.typed.stat.Return) then
    return node

  elseif node:is(ast.typed.stat.Break) then
    return node

  elseif node:is(ast.typed.stat.Assignment) then
    return node

  elseif node:is(ast.typed.stat.Reduce) then
    return node

  elseif node:is(ast.typed.stat.Expr) then
    return node

  elseif node:is(ast.typed.stat.BeginTrace) then
    return node

  elseif node:is(ast.typed.stat.EndTrace) then
    return node

  elseif node:is(ast.typed.stat.MapRegions) then
    return node

  elseif node:is(ast.typed.stat.UnmapRegions) then
    return node

  elseif node:is(ast.typed.stat.RawDelete) then
    return node

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function cudaize_loops.top_task(node)
  local body = cudaize_loops.block(node.body)

  return node { body = body }
end

function cudaize_loops.top(node)
  if node:is(ast.typed.top.Task) then
    return cudaize_loops.top_task(node)
  else
    return node
  end
end

function cudaize_loops.entry(node)
  return cudaize_loops.top(node)
end

cudaize = {}
function cudaize.stat_for_list(cx, node, reduction)
  assert(node:is(ast.typed.stat.ForList))
  assert(reduction.lhs[1]:is(ast.typed.expr.ID))

  local stats = terralib.newlist()

  -- var N = r.ispace.volume
  local region_expr = node.value
  local ispace_field_access = ast_util.mk_expr_field_access(region_expr, "ispace", ispace(ptr))
  local volume_field_access = ast_util.mk_expr_field_access(ispace_field_access, "volume", int64)
  local N_symbol = std.newsymbol(int64, "N")
  stats:insert(ast_util.mk_stat_var(N_symbol, int64, volume_field_access))

  local N_expr = ast_util.mk_expr_id(N_symbol, int64)

  -- var tmp[] = calloc(N, sizeof(type))
  local reduction_type = std.as_read(reduction.lhs[1].expr_type)
  local tmp_symbol = std.newsymbol(&reduction_type, "tmp")
  local calloc_args = terralib.newlist()
  calloc_args:insert(N_expr)
  calloc_args:insert(ast_util.mk_expr_constant(terralib.sizeof(reduction_type), uint))
  local calloc_expr = ast_util.mk_expr_call(c.calloc, calloc_args)
  local calloc_casted = ast_util.mk_expr_cast(&reduction_type, calloc_expr)
  stats:insert(ast_util.mk_stat_var(tmp_symbol, &reduction_type, calloc_casted))

  -- for i = 0, N do tmp[i] = 0 end
  local i_symbol = std.newsymbol(uint, "i")

  local tmp_range = terralib.newlist()
  tmp_range:insert(ast_util.mk_expr_constant(0, uint))
  tmp_range:insert(N_expr)

  local tmp_array_expr = ast_util.mk_expr_id(tmp_symbol, std.rawref(&tmp_symbol:gettype()))
  local i_expr = ast_util.mk_expr_id(i_symbol, std.rawref(&uint))
  local tmp_index_access = ast_util.mk_expr_index_access(tmp_array_expr, i_expr, std.rawref(&reduction_type))
  local zero_expr = ast_util.mk_expr_cast(reduction_type, ast_util.mk_expr_constant(0, reduction_type))
  local init_block = ast_util.mk_block(ast_util.mk_stat_assignment(tmp_index_access, zero_expr))
  stats:insert(ast_util.mk_stat_for_num(i_symbol, tmp_range, init_block))

  -- var idx = 0
  local idx_symbol = std.newsymbol(uint, "idx")
  stats:insert(ast_util.mk_stat_var(idx_symbol, nil, ast_util.mk_expr_constant(0, uint)))

  -- modify loop body to replace the reduction with tmp[idx] = e.f, idx++
  local block = cudaize_loops.block(node.block)

  for i, stat in ipairs(block.stats) do
    if (stat == reduction) then
      local idx_expr = ast_util.mk_expr_id(idx_symbol, std.rawref(&uint))
      local idx_lhs = ast_util.mk_expr_index_access(tmp_array_expr, idx_expr, std.rawref(&reduction_type))

      table.insert(block.stats, i, ast_util.mk_stat_reduce("+", idx_expr, ast_util.mk_expr_constant(1, uint)))
      table.insert(block.stats, i, ast_util.mk_stat_assignment(idx_lhs, reduction.rhs[1]))
      table.remove(block.stats, i+2)
      break
    end
  end
  stats:insert(ast_util.mk_stat_for_list(node.symbol, node.value, block))

  -- for i = 0, N do x += tmp[i] end
  local final_block = ast_util.mk_empty_block()
  final_block.stats:insert(ast_util.mk_stat_assignment(reduction.lhs[1], tmp_index_access))
  stats:insert(ast_util.mk_stat_for_num(i_symbol, tmp_range, final_block))

  return stats
end

cudaize_loops.pass_name = "cudaize_loops"

return cudaize_loops
